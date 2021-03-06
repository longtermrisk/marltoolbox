import copy
import logging
from typing import List, Union, Optional, Dict, Tuple, Iterable

import torch
from ray import rllib
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType

from marltoolbox.utils import log

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "nested_policies": [
        # You need to provide the policy class for every nested Policies
        {"Policy_class": None, "config_update": {}},
        {"Policy_class": None, "config_update": {}},
    ],
}


class HierarchicalTorchPolicy(rllib.policy.TorchPolicy):
    INITIALLY_ACTIVE_ALGO = 0

    def __init__(
        self,
        observation_space,
        action_space,
        config,
        after_init_nested=None,
        **kwargs,
    ):
        self.algorithms = []
        self.config = config
        # TODO clean this
        for nested_config in self.config["nested_policies"]:
            updated_config = copy.deepcopy(config)
            updated_config.update(nested_config["config_update"])
            if nested_config["Policy_class"] is None:
                raise ValueError(
                    f"You must specify the classes for the nested Policies "
                    f'in config["nested_config"]["Policy_class"] '
                    f'current value is {nested_config["Policy_class"]}.'
                )
            Policy = nested_config["Policy_class"]
            policy = Policy(
                observation_space, action_space, updated_config, **kwargs
            )
            if after_init_nested is not None:
                after_init_nested(policy)
            self.algorithms.append(policy)

        self.active_algo_idx = self.INITIALLY_ACTIVE_ALGO

        super().__init__(
            observation_space,
            action_space,
            config,
            model=self.model,
            loss=None,
            action_distribution_class=self.dist_class,
            **kwargs,
        )

        for algo in self.algorithms:
            algo.model = algo.model.to(self.device)

        self._to_log = {}
        self._already_printed_warnings = []

    def __getattribute__(self, attr):
        """
        Here we try to fallback to attributes in the active policy.
        """
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as initial:
            try:
                msg = (
                    f"{self} doesn't have the attribute {attr}. "
                    f"Defaulting to the same attribute of "
                    f"the active nested policy "
                    f"{self.algorithms[self.active_algo_idx]}. No more "
                    f"printing this message."
                )
                logger.info(msg)
                # from warnings import warn
                # warn(msg)
                return object.__getattribute__(
                    self.algorithms[self.active_algo_idx], attr
                )
            except AttributeError as secondary:
                raise type(initial)(f"{initial.args} and {secondary.args}")

    @property
    def to_log(self):
        to_log = {
            "meta_policy": self._to_log,
            "nested_policy": {
                f"policy_{algo_idx}": algo.to_log
                for algo_idx, algo in enumerate(self.algorithms)
                if hasattr(algo, "to_log")
            },
        }
        return to_log

    @to_log.setter
    def to_log(self, value):
        if value == {}:
            for algo in self.algorithms:
                if hasattr(algo, "to_log"):
                    algo.to_log = {}
                # setattr(algo, "to_log", {})

        self._to_log = value

    @property
    def model(self):
        return self.algorithms[self.active_algo_idx].model

    @model.setter
    def model(self, value):
        self.algorithms[self.active_algo_idx].model = value

    @property
    def dist_class(self):
        return self.algorithms[self.active_algo_idx].dist_class

    @dist_class.setter
    def dist_class(self, value):
        self.algorithms[self.active_algo_idx].dist_class = value

    @property
    def global_timestep(self):
        return self.algorithms[self.active_algo_idx].global_timestep

    @global_timestep.setter
    def global_timestep(self, value):
        for algo in self.algorithms:
            algo.global_timestep = value

    def on_global_var_update(self, global_vars):
        for algo in self.algorithms:
            algo.on_global_var_update(global_vars)

    @property
    def update_target(self):
        def nested_update_target():
            for algo in self.algorithms:
                if "update_target" in algo.__dict__:
                    algo.update_target()

        return nested_update_target

    def get_weights(self):
        return {
            self.nested_key(i): algo.get_weights()
            for i, algo in enumerate(self.algorithms)
        }

    def set_weights(self, weights):
        for i, algo in enumerate(self.algorithms):
            algo.set_weights(weights[self.nested_key(i)])

    def nested_key(self, i):
        return f"nested_{i}"

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

        actions, state_out, extra_fetches = self.algorithms[
            self.active_algo_idx
        ].compute_actions(obs_batch)

        return actions, state_out, extra_fetches

    def learn_on_batch(self, samples: SampleBatch):
        self._update_lr_in_all_optimizers()
        stats = self._learn_on_batch(samples)
        self._log_learning_rates()
        return stats

    def _learn_on_batch(self, samples: SampleBatch):
        raise NotImplementedError()

    def _update_lr_in_all_optimizers(self):
        self.optimizer()

    def _init_log_learn_on_batch(self, algo_idx: Iterable = None):
        if algo_idx is not None:
            for policy_n in algo_idx:
                self._to_log[f"learn_on_batch_algo{policy_n}"] = 0.0
        else:
            for policy_n, _ in enumerate(self.algorithms):
                self._to_log[f"learn_on_batch_algo{policy_n}"] = 0.0

    def optimizer(
        self,
    ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        """Custom the local PyTorch optimizer(s) to use.

        Returns:
            Union[List[torch.optim.Optimizer], torch.optim.Optimizer]:
                The local PyTorch optimizer(s) to use for this Policy.
        """

        # TODO find a clean solution to update the LR when using a LearningRateSchedule
        # TODO this will probably no more be needed when moving to RLLib>1.0.0
        for algo in self.algorithms:
            if hasattr(algo, "cur_lr"):
                for opt in algo._optimizers:
                    for p in opt.param_groups:
                        p["lr"] = algo.cur_lr

        all_optimizers = []
        for algo_n, algo in enumerate(self.algorithms):
            opt = algo.optimizer()
            all_optimizers.extend(opt)
        return all_optimizers

    # TODO Move this in helper functions

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return self.algorithms[self.active_algo_idx].postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )

    def _log_learning_rates(self):
        """
        Use to log LR to check that they are really updated as configured.
        """
        for algo_idx, algo in enumerate(self.algorithms):
            self.to_log[f"algo{algo_idx}"] = log._log_learning_rate(algo)

    def set_state(self, state: object) -> None:
        state = state.copy()  # shallow copy
        # Set optimizer vars first.
        optimizer_vars = state.pop("_optimizer_variables", None)
        if optimizer_vars:
            print("self", self)
            assert len(optimizer_vars) == len(
                self._optimizers
            ), f"{len(optimizer_vars)} {len(self._optimizers)}"
            for o, s in zip(self._optimizers, optimizer_vars):
                o.load_state_dict(s)
        # Then the Policy's (NN) weights.
        super().set_state(state)
