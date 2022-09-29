"""
PyTorch policy class used for PPO.
"""
import logging
from typing import Union, List

import gym
import ray
from ray.rllib.agents.ppo import PPOTorchPolicy

# from ray.rllib.agents.ppo.ppo_torch_policy import (
#     ValueNetworkMixin,
#     KLCoeffMixin,
# )
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, TorchPolicy
from ray.rllib.utils import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TrainerConfigDict

from marltoolbox.utils.policy import MyLearningRateSchedule

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


# def setup_mixins(
#     policy: Policy,
#     obs_space: gym.spaces.Space,
#     action_space: gym.spaces.Space,
#     config: TrainerConfigDict,
# ) -> None:
#     """Call all mixin classes' constructors before PPOPolicy initialization.
#
#     Args:
#         policy (Policy): The Policy object.
#         obs_space (gym.spaces.Space): The Policy's observation space.
#         action_space (gym.spaces.Space): The Policy's action space.
#         config (TrainerConfigDict): The Policy's config.
#     """
#     ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
#     KLCoeffMixin.__init__(policy, config)
#     EntropyCoeffSchedule.__init__(
#         policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
#     )
#     MyLearningRateSchedule.__init__(
#         policy, config["lr"], config["lr_schedule"]
#     )
#
#
# MyPPOTorchPolicy = PPOTorchPolicy.with_updates(
#     before_loss_init=setup_mixins,
#     mixins=[
#         MyLearningRateSchedule,
#         EntropyCoeffSchedule,
#         KLCoeffMixin,
#         ValueNetworkMixin,
#     ],
# )
class MyPPOTorchPolicy(PPOTorchPolicy, MyLearningRateSchedule):
    # def __init__(self, observation_space, action_space, config):
    #     config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
    #     setup_config(self, observation_space, action_space, config)
    #
    #     TorchPolicy.__init__(
    #         self,
    #         observation_space,
    #         action_space,
    #         config,
    #         max_seq_len=config["model"]["max_seq_len"],
    #     )
    #
    #     EntropyCoeffSchedule.__init__(
    #         self, config["entropy_coeff"], config["entropy_coeff_schedule"]
    #     )
    #     MyLearningRateSchedule.__init__(
    #         self, config["lr"], config["lr_schedule"]
    #     )
    #
    #     # The current KL value (as python float).
    #     self.kl_coeff = self.config["kl_coeff"]
    #     # Constant target value.
    #     self.kl_target = self.config["kl_target"]
    #
    #     # TODO: Don't require users to call this manually.
    #     self._initialize_loss_from_dummy_batch()

    @DeveloperAPI
    def optimizer(
        self,
    ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        """Custom the local PyTorch optimizer(s) to use.

        Returns:
            The local PyTorch optimizer(s) to use for this Policy.
        """
        if hasattr(self, "config"):
            optimizers = [
                self.config["optimizer_class"](
                    self.model.parameters(), lr=self.config["lr"]
                )
            ]
        else:
            raise ValueError()
            # optimizers = [torch.optim.Adam(self.model.parameters())]
        if getattr(self, "exploration", None):
            optimizers = self.exploration.get_exploration_optimizer(optimizers)
        return optimizers
