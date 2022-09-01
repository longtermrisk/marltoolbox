import torch
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict


def sgd_optimizer_dqn(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    if not hasattr(policy, "q_func_vars"):
        policy.q_func_vars = policy.model.variables()
    return sgd_optimizer(policy, config, policy.q_func_vars)


def sgd_optimizer_spl(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    return sgd_optimizer(policy, config, policy.model.parameters())


def sgd_optimizer(policy: Policy, config: TrainerConfigDict, parameters):
    return torch.optim.SGD(
        parameters,
        lr=policy.cur_lr,
        momentum=config["optimizer"]["sgd_momentum"],
        weight_decay=config["env_config"].get("weight_decay", 1e-4),
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

    # assert len(list(config["optimizer"].keys())) == 0

    return torch.optim.Adam(
        policy.q_func_vars,
        lr=policy.cur_lr,
        eps=config["adam_epsilon"],
        betas=betas,
    )


def adam_optimizer_spl(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    if "betas" in config["optimizer"].keys():
        betas = config["optimizer"].pop("betas")
    else:
        betas = (0.9, 0.999)
    return torch.optim.Adam(
        policy.model.parameters(),
        lr=policy.cur_lr,
        eps=config["adam_epsilon"],
        betas=betas,
    )
