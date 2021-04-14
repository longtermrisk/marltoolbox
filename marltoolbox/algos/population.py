import copy
import random
from typing import List, Union, Optional, Dict, Tuple

from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from marltoolbox.algos import hierarchical
from marltoolbox.utils import miscellaneous, restore

DEFAULT_CONFIG = merge_dicts(
    hierarchical.DEFAULT_CONFIG,
    {
        # To configure
        "policy_checkpoints": [],
        "policy_id_to_load": None,
        "nested_policies": None,
        "freeze_algo": True,
        "use_random_algo": True,
        "use_algo_in_order": False,
        "switch_of_algo_every_n_epi": 1,
    },
)


# TODO to test with TF (only tested with PyTorch)
class PopulationOfIdenticalAlgo(hierarchical.HierarchicalTorchPolicy):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)

        assert len(self.algorithms) == 1
        self.active_algo_idx = 0

        self.freeze_algo = config["freeze_algo"]
        self.policy_id_to_load = config["policy_id_to_load"]
        self.use_random_algo = config["use_random_algo"]
        self.use_algo_in_order = config["use_algo_in_order"]
        self.policy_checkpoints = config["policy_checkpoints"]
        self.switch_of_algo_every_n_epi = config["switch_of_algo_every_n_epi"]
        self.switch_of_algo_counter = self.switch_of_algo_every_n_epi
        if callable(self.policy_checkpoints):
            self.policy_checkpoints = self.policy_checkpoints(self.config)

        assert self.use_random_algo or self.use_algo_in_order
        assert not self.use_random_algo or not self.use_algo_in_order
        assert isinstance(self.switch_of_algo_every_n_epi, int)

        self.set_algo_to_use()

    def set_algo_to_use(self):
        """
        Called by a callback at the start of every episode.
        """

        if self.switch_of_algo_counter == self.switch_of_algo_every_n_epi:
            self.switch_of_algo_counter = 0
            self._set_algo_to_use()
        else:
            self.switch_of_algo_counter += 1

    def _set_algo_to_use(self):
        self._select_algo_idx_to_use()
        self._load_checkpoint()
        if self.freeze_algo:
            self._freeze_nested_algo_in_use()

    def _select_algo_idx_to_use(self):
        if self.use_random_algo:
            self._use_random_algo()
        elif self.use_algo_in_order:
            self._use_next_algo()
        else:
            raise ValueError()

    def _use_random_algo(self):
        self.active_checkpoint_idx = random.randint(
            0, len(self.policy_checkpoints) - 1
        )

    def _use_next_algo(self):
        self.active_checkpoint_idx += 1
        if self.active_checkpoint_idx == len(self.policy_checkpoints):
            self.active_checkpoint_idx = 0

    def _load_checkpoint(self):
        restore.load_one_policy_checkpoint(
            policy_id=self.policy_id_to_load,
            policy=self.algorithms[self.active_algo_idx],
            checkpoint_path=self.policy_checkpoints[
                self.active_checkpoint_idx
            ],
            using_Tune_class=hasattr(
                self.algorithms[self.active_algo_idx], "tune_config"
            ),
        )

    def _freeze_nested_algo_in_use(self):
        if hasattr(self.algorithms[self.active_algo_idx].model, "eval"):
            self.algorithms[self.active_algo_idx].model.eval()

    @override(hierarchical.HierarchicalTorchPolicy)
    def get_weights(self):
        return self.algorithms[self.active_algo_idx].get_weights()

    @override(hierarchical.HierarchicalTorchPolicy)
    def set_weights(self, weights):
        self.algorithms[self.active_algo_idx].set_weights(weights)

    @override(hierarchical.HierarchicalTorchPolicy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorType], TensorType] = None,
        prev_reward_batch: Union[List[TensorType], TensorType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["MultiAgentEpisode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        return self.algorithms[self.active_algo_idx].compute_actions(obs_batch)

    @override(hierarchical.HierarchicalTorchPolicy)
    def learn_on_batch(self, samples: SampleBatch):
        if not self.freeze_algo:
            # TODO maybe need to call optimizer to update the LR of the nested optimizers
            learner_stats = {"learner_stats": {}}
            for i, algo in enumerate(self.algorithms):
                stats = self.algorithms[self.active_algo_idx].learn_on_batch(
                    samples
                )
                learner_stats["learner_stats"][f"algo_{i}"] = stats
            return learner_stats
        else:
            raise NotImplementedError(
                "Policies in PopulationOfAlgo are freezed thus "
                "PopulationOfAlgo.learn_on_batch should not be called"
            )

    @property
    @override(hierarchical.HierarchicalTorchPolicy)
    def to_log(self):
        return self.algorithms[self.active_algo_idx].to_log

    @to_log.setter
    @override(hierarchical.HierarchicalTorchPolicy)
    def to_log(self, value):
        self.algorithms[self.active_algo_idx].to_log = value

    @override(hierarchical.HierarchicalTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return self.algorithms[self.active_algo_idx].postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )

    def on_episode_start(
        self,
        *args,
        **kwargs,
    ):
        self.set_algo_to_use()


def modify_config_to_use_population(
    config: dict, population_policy_id: str, opponents_checkpoints: list
):
    population_policy = copy.deepcopy(
        list(config["multiagent"]["policies"][population_policy_id])
    )

    population_policy = _convert_to_population_policy(
        population_policy, config, population_policy_id, opponents_checkpoints
    )

    miscellaneous.overwrite_config(
        dict_=config,
        key=f"multiagent.policies.{population_policy_id}",
        value=population_policy,
    )
    return config


def _convert_to_population_policy(
    population_policy, config, opponent_policy_id, opponents_checkpoints
):
    population_policy[0] = PopulationOfIdenticalAlgo
    population_policy[3].update(DEFAULT_CONFIG)
    population_policy[3]["policy_id_to_load"] = opponent_policy_id
    population_policy[3]["nested_policies"] = [
        # You need to provide the policy class for every nested Policies
        {
            "Policy_class": copy.deepcopy(
                config["multiagent"]["policies"][opponent_policy_id][0]
            ),
            "config_update": copy.deepcopy(
                config["multiagent"]["policies"][opponent_policy_id][3]
            ),
        },
    ]
    population_policy[3]["policy_checkpoints"] = opponents_checkpoints
    return population_policy
