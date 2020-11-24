from ray.rllib.agents.trainer import with_common_config
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy import Policy
from ray.rllib.policy import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from typing import Dict, List, Type, Union
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

torch, _ = try_import_torch()

SPL_DEFAULT_CONFIG = with_common_config({
    "learn_action": True,
    "learn_reward": False,
    "loss_fn": torch.nn.CrossEntropyLoss(
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction='mean')
})

def spl_torch_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
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
    predictions = dist_class(dist_inputs, model)

    targets = []
    if policy.config["learn_action"]:
        targets.append(train_batch[SampleBatch.ACTIONS])
    if policy.config["learn_reward"]:
        targets.append(train_batch[SampleBatch.REWARDS])
    assert len(targets) > 0
    targets = torch.cat(targets, dim=0)


    # Save the loss in the policy object for the spl_stats below.
    policy.spl_loss = policy.config["loss_fn"](predictions.dist.probs, targets)
    policy.entropy = predictions.dist.entropy().mean()

    return policy.spl_loss


def spl_stats(policy: Policy,
                   train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Returns the calculated loss in a stats dict.

    Args:
        policy (Policy): The Policy object.
        train_batch (SampleBatch): The data used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """

    return {
        "cur_lr": policy.cur_lr,
        "entropy": policy.entropy,
        "err_policy_spl_loss": policy.spl_loss.item(),
    }

def adam_optimizer(policy: Policy,
                   config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.model.parameters(), lr=policy.cur_lr, eps=config["adam_epsilon"])

def setup_early_mixins(policy: Policy, obs_space, action_space,
                       config: TrainerConfigDict) -> None:
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


SPLTorchPolicy = build_torch_policy(
    name="SPLTorchPolicy",
    get_default_config=lambda: SPL_DEFAULT_CONFIG,
    loss_fn=spl_torch_loss,
    stats_fn=spl_stats,
    optimizer_fn=adam_optimizer,
    before_init=setup_early_mixins,
    mixins=[
        LearningRateSchedule,
    ])
