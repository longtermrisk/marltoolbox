from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict

import torch


def sgd_optimizer_dqn(policy: Policy,
                      config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.q_func_vars, lr=policy.cur_lr, momentum=config["sgd_momentum"])


def sgd_optimizer_spl(policy: Policy,
                      config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.model.parameters(),
        lr=policy.cur_lr,
        momentum=config["sgd_momentum"])
