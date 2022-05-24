import torch
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict


def sgd_optimizer_dqn(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.q_func_vars,
        lr=policy.cur_lr,
        momentum=config["optimizer"]["sgd_momentum"],
    )


def sgd_optimizer_spl(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.model.parameters(),
        lr=policy.cur_lr,
        momentum=config["optimizer"]["sgd_momentum"],
    )


def adam_optimizer_dqn(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    # By this time, the models have been moved to the GPU - if any - and we
    # can define our optimizers using the correct CUDA variables.
    if not hasattr(policy, "q_func_vars"):
        policy.q_func_vars = policy.model.variables()

    if "betas" in config["optimizer"].keys():
        betas = config["optimizer"].pop("betas")
    else:
        betas = (0.9, 0.999)

    assert len(list(config["optimizer"].keys())) == 0

    return torch.optim.Adam(
        policy.q_func_vars,
        lr=policy.cur_lr,
        eps=config["adam_epsilon"],
        betas=betas,
    )


def adam_optimizer_spl(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    return torch.optim.Adam(
        policy.model.parameters(), lr=policy.cur_lr, eps=config["adam_epsilon"]
    )
