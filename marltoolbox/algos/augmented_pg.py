from typing import Union, Type, List, Dict

import torch
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.pg import PGTrainer, PGTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.typing import TrainerConfigDict, TensorType

from marltoolbox.utils import log


def my_pg_torch_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """The basic policy gradients loss function.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Pass the training data through our model to get distribution parameters.
    dist_inputs, _ = model(train_batch)

    # Create an action distribution object.
    action_dist = dist_class(dist_inputs, model)
    curr_entropy = action_dist.entropy()

    # Calculate the vanilla PG loss based on:
    # L = -E[ log(pi(a|s)) * A]
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])

    # Final policy loss.
    policy_loss = -torch.mean(log_probs * train_batch[Postprocessing.ADVANTAGES])
    entropy_loss = -policy.entropy_coeff * curr_entropy.mean()

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["policy_loss"] = policy_loss
    model.tower_stats["entropy_loss"] = entropy_loss

    return policy_loss + entropy_loss


def setup_early_mixins(
    policy: Policy, obs_space, action_space, config: TrainerConfigDict
) -> None:
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )


def sgd_optimizer(policy: Policy, config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(policy.model.parameters(), lr=policy.cur_lr)


def my_pg_loss_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Returns the calculated loss in a stats dict.

    Args:
        policy (Policy): The Policy object.
        train_batch (SampleBatch): The data used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """

    return {
        "policy_loss": torch.mean(torch.stack(policy.get_tower_stats("policy_loss"))),
        "entropy_loss": torch.mean(torch.stack(policy.get_tower_stats("entropy_loss"))),
    }


MyPGTorchPolicy = PGTorchPolicy.with_updates(
    optimizer_fn=sgd_optimizer,
    before_init=setup_early_mixins,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule],
    stats_fn=log.augment_stats_fn_wt_additionnal_logs(my_pg_loss_stats),
    loss_fn=my_pg_torch_loss,
)


class MyPGTrainer(PGTrainer):
    _allow_unknown_configs = True
