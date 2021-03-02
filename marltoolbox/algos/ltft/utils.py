from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, build_q_stats
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TrainerConfigDict

torch, nn = try_import_torch()

from marltoolbox.utils import log

def sgd_optimizer_dqn(policy: Policy, config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.q_func_vars, lr=policy.cur_lr, momentum=config["sgd_momentum"])


def sgd_optimizer_spl(policy: Policy, config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.model.parameters(), lr=policy.cur_lr, momentum=config["sgd_momentum"])


MyDQNTorchPolicy = DQNTorchPolicy.with_updates(
    optimizer_fn=sgd_optimizer_dqn,
    stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats))