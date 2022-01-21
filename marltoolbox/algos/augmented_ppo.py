"""
PyTorch policy class used for PPO.
"""
import logging

import gym
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import (
    ValueNetworkMixin,
    KLCoeffMixin,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TrainerConfigDict

from marltoolbox.utils.policy import MyLearningRateSchedule

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


def setup_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> None:
    """Call all mixin classes' constructors before PPOPolicy initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    MyLearningRateSchedule.__init__(
        policy, config["lr"], config["lr_schedule"]
    )


MyPPOTorchPolicy = PPOTorchPolicy.with_updates(
    before_loss_init=setup_mixins,
    mixins=[
        MyLearningRateSchedule,
        EntropyCoeffSchedule,
        KLCoeffMixin,
        ValueNetworkMixin,
    ],
)
