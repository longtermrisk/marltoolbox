"""PyTorch policy class used for R2D2."""

from ray.rllib.agents.dqn.dqn_torch_policy import ComputeTDErrorMixin
from ray.rllib.agents.dqn.r2d2_torch_policy import (
    R2D2TorchPolicy,
    r2d2_loss,
    build_q_stats,
)
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.policy.policy import Policy

# from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.typing import TrainerConfigDict

from marltoolbox.utils import log, optimizers, policy


def setup_early_mixins(
    policy: Policy, obs_space, action_space, config: TrainerConfigDict
) -> None:
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


MyR2D2TorchPolicy = R2D2TorchPolicy.with_updates(
    optimizer_fn=optimizers.sgd_optimizer_dqn,
    loss_fn=r2d2_loss,
    stats_fn=log.augment_stats_fn_wt_additionnal_logs(build_q_stats),
    before_init=setup_early_mixins,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        policy.MyLearningRateSchedule,
    ],
)

MyAdamR2D2TorchPolicy = MyR2D2TorchPolicy.with_updates(
    optimizer_fn=optimizers.adam_optimizer_dqn,
)
MyAdamVanillaR2D2TorchPolicy = R2D2TorchPolicy.with_updates(
    optimizer_fn=optimizers.adam_optimizer_dqn,
)

MySGDVanillaR2D2TorchPolicy = R2D2TorchPolicy.with_updates(
    optimizer_fn=optimizers.sgd_optimizer_dqn,
)
