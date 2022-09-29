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
    if "weight_decay" in config["optimizer"]:
        weight_decay = config["optimizer"]["weight_decay"]
    elif "weight_decay" in config["env_config"]:
        weight_decay = config["env_config"]["weight_decay"]
    else:
        weight_decay = 1e-4

    if "momentum" in config["optimizer"]:
        momentum = config["optimizer"]["momentum"]
    elif "momentum" in config["env_config"]:
        momentum = config["env_config"]["momentum"]
    else:
        momentum = 0.0

    return torch.optim.SGD(
        parameters,
        lr=policy.cur_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def adam_optimizer_dqn(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    # By this time, the models have been moved to the GPU - if any - and we
    # can define our optimizers using the correct CUDA variables.
    if not hasattr(policy, "q_func_vars"):
        policy.q_func_vars = policy.model.variables()

    return adam_optimizer(policy, config, policy.q_func_vars)


def adam_optimizer_spl(
    policy: Policy, config: TrainerConfigDict
) -> "torch.optim.Optimizer":
    return adam_optimizer(policy, config, policy.model.parameters())


def adam_optimizer(policy: Policy, config: TrainerConfigDict, parameters):
    if "betas" in config["optimizer"].keys():
        betas = config["optimizer"].pop("betas")
    elif "betas" in config["env_config"].keys():
        betas = config["env_config"]["env_config"]
    else:
        betas = (0.9, 0.999)

    if "weight_decay" in config["optimizer"]:
        weight_decay = config["optimizer"]["weight_decay"]
    elif "weight_decay" in config["env_config"]:
        weight_decay = config["env_config"]["weight_decay"]
    else:
        weight_decay = 0.0

    if "adam_epsilon" in config["optimizer"]:
        adam_epsilon = config["optimizer"]["adam_epsilon"]
    elif "adam_epsilon" in config["env_config"]:
        adam_epsilon = config["env_config"]["adam_epsilon"]
    else:
        adam_epsilon = 1e-08

    return torch.optim.Adam(
        parameters,
        lr=policy.cur_lr,
        eps=adam_epsilon,
        betas=betas,
        weight_decay=weight_decay,
    )
