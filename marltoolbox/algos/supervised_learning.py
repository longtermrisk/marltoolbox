from typing import Dict, List, Type, Union

import torch
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy import Policy
from ray.rllib.policy import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

from marltoolbox.utils import log, optimizers, policy

SPL_DEFAULT_CONFIG = with_common_config(
    {
        "learn_action": True,
        "learn_reward": False,
        "loss_fn": torch.nn.CrossEntropyLoss(),
    }
)


def spl_torch_loss(
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
    dist_inputs, _ = model.from_batch(train_batch)
    # Create an action distribution object.
    action_dist = dist_class(dist_inputs, model)
    if policy.config["explore"]:
        # Adding that because of a bug in TorchCategorical
        #  which modify dist_inputs through action_dist:
        _, _ = policy.exploration.get_exploration_action(
            action_distribution=action_dist,
            timestep=policy.global_timestep,
            explore=policy.config["explore"],
        )
        action_dist = dist_class(dist_inputs, policy.model)

    targets = []
    if policy.config["learn_action"]:
        targets.append(train_batch[SampleBatch.ACTIONS])
    if policy.config["learn_reward"]:
        targets.append(train_batch[SampleBatch.REWARDS])
    assert len(targets) > 0, (
        f"In config, use learn_action=True and/or "
        f"learn_reward=True to specify which target to "
        f"use in supervised learning"
    )
    targets = torch.cat(targets, dim=0)

    # Save the loss in the policy object for the spl_stats below.
    policy.spl_loss = policy.config["loss_fn"](action_dist.dist.probs, targets)
    policy.entropy = action_dist.dist.entropy().mean()

    return policy.spl_loss


def spl_stats(
    policy: Policy, train_batch: SampleBatch
) -> Dict[str, TensorType]:
    """Returns the calculated loss in a stats dict.

    Args:
        policy (Policy): The Policy object.
        train_batch (SampleBatch): The data used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """

    return {
        "cur_lr": policy.cur_lr,
        "entropy_avg": policy.entropy,
        "err_policy_spl_loss": policy.spl_loss.item(),
    }


def setup_early_mixins(
    policy: Policy, obs_space, action_space, config: TrainerConfigDict
) -> None:
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


SPLTorchPolicy = build_torch_policy(
    name="SPLTorchPolicy",
    get_default_config=lambda: SPL_DEFAULT_CONFIG,
    loss_fn=spl_torch_loss,
    stats_fn=spl_stats,
    optimizer_fn=optimizers.adam_optimizer_spl,
    before_init=setup_early_mixins,
    mixins=[
        LearningRateSchedule,
    ],
)

MySPLTorchPolicy = SPLTorchPolicy.with_updates(
    optimizer_fn=optimizers.sgd_optimizer_spl,
    stats_fn=log.augment_stats_fn_wt_additionnal_logs(spl_stats),
    before_init=policy.my_setup_early_mixins,
    mixins=[
        policy.MyLearningRateSchedule,
    ],
)

MyAdamSPLTorchPolicy = SPLTorchPolicy.with_updates(
    optimizer_fn=optimizers.adam_optimizer_spl,
)
