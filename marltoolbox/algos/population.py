import copy

import random
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.typing import PolicyID
from ray.rllib.utils.typing import TensorType
from typing import List, Union, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

from marltoolbox.algos import hierarchical
from marltoolbox.utils import miscellaneous, restore

DEFAULT_CONFIG_UPDATE = merge_dicts(
    hierarchical.DEFAULT_CONFIG,
    {
        # To configure
        "policy_checkpoints": [],
        "policy_id_to_load": None,
        'nested_policies': None,

        "freeze_algo": True,
        "use_random_algo": True,
        "use_algo_in_order": False,
    }
)


class PopulationOfIdenticalAlgoCallBacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        """ Used to swtich between agents/checkpoints in the population"""

        for policy_id, policy in policies.items():
            if isinstance(policy, PopulationOfIdenticalAlgo):
                policy.set_algo_to_use()


# TODO not tested with TF
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
        if callable(self.policy_checkpoints):
            self.policy_checkpoints = self.policy_checkpoints(self.config)

        self.set_algo_to_use()

    def set_algo_to_use(self):
        assert self.use_random_algo or self.use_algo_in_order
        assert not self.use_random_algo or not self.use_algo_in_order

        if self.use_random_algo:
            self._use_random_algo()
        elif self.use_algo_in_order:
            self._use_next_algo()
        else:
            raise ValueError()

        self._load_checkpoint()

        if self.freeze_algo:
            if hasattr(self.algorithms[0].model, "eval"):
                self.algorithms[0].model.eval()

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
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        return self.algorithms[0].compute_actions(obs_batch)

    def learn_on_batch(self, samples: SampleBatch):
        if not self.freeze_algo:
            # TODO maybe need to call optimizer to update the LR of the nested optimizers
            learner_stats = {"learner_stats": {}}
            for i, algo in enumerate(self.algorithms):
                stats = self.algorithms[0].learn_on_batch(samples)
                learner_stats["learner_stats"][f"algo_{i}"] = stats
            return learner_stats
        else:
            raise NotImplementedError("Policies in PopulationOfAlgo are freezed thus "
                                      "PopulationOfAlgo.learn_on_batch should not be called")

    def _use_random_algo(self):
        self.active_checkpoint_idx = random.randint(0, len(self.policy_checkpoints) - 1)

    def _use_next_algo(self):
        self.active_checkpoint_idx += 1
        if self.active_checkpoint_idx == len(self.policy_checkpoints):
            self.active_checkpoint_idx = 0

    def _load_checkpoint(self):
        # print("self.policy_checkpoints[self.active_checkpoint_idx]",
        #       self.policy_checkpoints[self.active_checkpoint_idx])
        restore.load_one_policy_checkpoint(policy_id=self.policy_id_to_load, policy=self.algorithms[0],
                                           checkpoint_path=self.policy_checkpoints[self.active_checkpoint_idx],
                                           using_Tune_class=hasattr(self.algorithms[0], "tune_config"))

    def get_weights(self):
        return self.algorithms[0].get_weights()

    def set_weights(self, weights):
        self.algorithms[0].set_weights(weights)


def modify_config_to_use_population(config: dict, opponent_policy_id: str, opponents_checkpoints: list):
    population_policy = copy.deepcopy(list(config["multiagent"]["policies"][opponent_policy_id]))

    population_policy = _convert_to_population_policy(population_policy, config,
                                                      opponent_policy_id, opponents_checkpoints)

    miscellaneous.overwrite_config(dict_=config,
                                   key=f"multiagent.policies.{opponent_policy_id}", value=population_policy)
    return config


def _convert_to_population_policy(population_policy, config, opponent_policy_id, opponents_checkpoints):
    population_policy[0] = PopulationOfIdenticalAlgo
    population_policy[3].update(DEFAULT_CONFIG_UPDATE)
    population_policy[3]["policy_id_to_load"] = opponent_policy_id
    population_policy[3]["nested_policies"] = [
        # You need to provide the policy class for every nested Policies
        {"Policy_class": copy.deepcopy(config["multiagent"]["policies"][opponent_policy_id][0]),
         "config_update": copy.deepcopy(config["multiagent"]["policies"][opponent_policy_id][3])},
    ]
    population_policy[3]["policy_checkpoints"] = opponents_checkpoints
    return population_policy
