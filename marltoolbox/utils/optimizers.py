import torch
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict


def sgd_optimizer_dqn(policy: Policy,
                      config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.q_func_vars,
        lr=policy.cur_lr,
        momentum=config["optimizer"]["sgd_momentum"])


def sgd_optimizer_spl(policy: Policy,
                      config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.model.parameters(),
        lr=policy.cur_lr,
        momentum=config["optimizer"]["sgd_momentum"])


def adam_optimizer_dqn(policy: Policy,
                       config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.Adam(
        policy.q_func_vars,
        lr=policy.cur_lr,
        eps=config["adam_epsilon"])


def adam_optimizer_spl(policy: Policy,
                       config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.Adam(
        policy.model.parameters(),
        lr=policy.cur_lr,
        eps=config["adam_epsilon"])
