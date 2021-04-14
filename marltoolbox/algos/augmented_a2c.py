import torch
from ray.rllib.agents.a3c.a3c_torch_policy import (
    A3CTorchPolicy,
    ValueNetworkMixin,
)
from ray.rllib.agents.dqn.dqn_torch_policy import setup_early_mixins
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.typing import TrainerConfigDict


def sgd_optimizer(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    return torch.optim.SGD(policy.model.parameters(), lr=policy.cur_lr)


MyA2CTorchPolicy = A3CTorchPolicy.with_updates(
    optimizer_fn=sgd_optimizer,
    before_init=setup_early_mixins,
    mixins=[ValueNetworkMixin, LearningRateSchedule],
)
