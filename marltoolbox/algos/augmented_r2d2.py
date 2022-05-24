"""PyTorch policy class used for R2D2."""

import torch
import torch.nn.functional as F
from ray.rllib.agents.dqn import r2d2_torch_policy
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.agents.dqn.dqn_torch_policy import ComputeTDErrorMixin
from ray.rllib.agents.dqn.dqn_torch_policy import compute_q_values
from ray.rllib.agents.dqn.r2d2_torch_policy import (
    R2D2TorchPolicy,
    h_function,
    h_inverse,
)
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

# from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.rllib.utils.torch_utils import (
    FLOAT_MIN,
    huber_loss,
    sequence_mask,
    l2_loss,
)
from ray.rllib.utils.typing import TensorType
from typing import Dict

from marltoolbox.utils import log, optimizers, policy


def my_r2d2_loss(
    policy: Policy, model, _, train_batch: SampleBatch
) -> TensorType:
    """Constructs the loss for R2D2TorchPolicy.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        train_batch (SampleBatch): The training data.

    Returns:
        TensorType: A single loss tensor.
    """
    config = policy.config

    # Construct internal state inputs.
    i = 0
    state_batches = []
    while "state_in_{}".format(i) in train_batch:
        state_batches.append(train_batch["state_in_{}".format(i)])
        i += 1
    assert state_batches

    # Q-network evaluation (at t).
    q, _, _, _ = compute_q_values(
        policy,
        model,
        train_batch,
        state_batches=state_batches,
        seq_lens=train_batch.get("seq_lens"),
        explore=False,
        is_training=True,
    )

    # Target Q-network evaluation (at t+1).
    q_target, _, _, _ = compute_q_values(
        policy,
        policy.target_q_model,
        train_batch,
        state_batches=state_batches,
        seq_lens=train_batch.get("seq_lens"),
        explore=False,
        is_training=True,
    )

    # Only additions
    policy.last_q = q.clone().detach()
    policy.last_q_target = q_target.clone().detach()

    actions = train_batch[SampleBatch.ACTIONS].long()
    dones = train_batch[SampleBatch.DONES].float()
    rewards = train_batch[SampleBatch.REWARDS]
    weights = train_batch[PRIO_WEIGHTS]

    B = state_batches[0].shape[0]
    T = q.shape[0] // B

    # Q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(actions, policy.action_space.n)
    q_selected = torch.sum(
        torch.where(q > FLOAT_MIN, q, torch.tensor(0.0, device=policy.device))
        * one_hot_selection,
        1,
    )

    if config["double_q"]:
        best_actions = torch.argmax(q, dim=1)
    else:
        best_actions = torch.argmax(q_target, dim=1)

    best_actions_one_hot = F.one_hot(best_actions, policy.action_space.n)
    q_target_best = torch.sum(
        torch.where(
            q_target > FLOAT_MIN,
            q_target,
            torch.tensor(0.0, device=policy.device),
        )
        * best_actions_one_hot,
        dim=1,
    )

    if config["num_atoms"] > 1:
        raise ValueError("Distributional R2D2 not supported yet!")
    else:
        q_target_best_masked_tp1 = (1.0 - dones) * torch.cat(
            [q_target_best[1:], torch.tensor([0.0], device=policy.device)]
        )

        if config["use_h_function"]:
            h_inv = h_inverse(
                q_target_best_masked_tp1, config["h_function_epsilon"]
            )
            target = h_function(
                rewards + config["gamma"] ** config["n_step"] * h_inv,
                config["h_function_epsilon"],
            )
        else:
            target = (
                rewards
                + config["gamma"] ** config["n_step"] * q_target_best_masked_tp1
            )

        # Seq-mask all loss-related terms.
        seq_mask = sequence_mask(train_batch["seq_lens"], T)[:, :-1]
        # Mask away also the burn-in sequence at the beginning.
        burn_in = policy.config["burn_in"]
        if burn_in > 0 and burn_in < T:
            seq_mask[:, :burn_in] = False

        num_valid = torch.sum(seq_mask)

        def reduce_mean_valid(t):
            return torch.sum(t[seq_mask]) / num_valid

        # Make sure use the correct time indices:
        # Q(t) - [gamma * r + Q^(t+1)]
        q_selected = q_selected.reshape([B, T])[:, :-1]
        td_error = q_selected - target.reshape([B, T])[:, :-1].detach()
        td_error = td_error * seq_mask
        weights = weights.reshape([B, T])[:, :-1]

        return weights, td_error, q_selected, reduce_mean_valid


def my_r2d2_loss_wt_huber_loss(
    policy: Policy, model, _, train_batch: SampleBatch
) -> TensorType:

    weights, td_error, q_selected, reduce_mean_valid = my_r2d2_loss(
        policy, model, _, train_batch
    )

    policy._total_loss = reduce_mean_valid(weights * huber_loss(td_error))
    policy._td_error = td_error.reshape([-1])
    policy._loss_stats = {
        "mean_q": reduce_mean_valid(q_selected),
        "min_q": torch.min(q_selected),
        "max_q": torch.max(q_selected),
        "mean_td_error": reduce_mean_valid(td_error),
    }

    return policy._total_loss


def my_r2d2_loss_wtout_huber_loss(
    policy: Policy, model, _, train_batch: SampleBatch
) -> TensorType:

    weights, td_error, q_selected, reduce_mean_valid = my_r2d2_loss(
        policy, model, _, train_batch
    )

    policy._total_loss = reduce_mean_valid(weights * l2_loss(td_error))
    policy._td_error = td_error.reshape([-1])
    policy._loss_stats = {
        "mean_q": reduce_mean_valid(q_selected),
        "min_q": torch.min(q_selected),
        "max_q": torch.max(q_selected),
        "mean_td_error": reduce_mean_valid(td_error),
    }

    return policy._total_loss


def my_build_q_stats(policy: Policy, batch) -> Dict[str, TensorType]:
    q_stats = r2d2_torch_policy.build_q_stats(policy, batch)

    entropy_avg, _ = log.compute_entropy_from_raw_q_values(
        policy, policy.last_q.clone()
    )
    q_stats.update(
        {
            "entropy_avg": entropy_avg,
        }
    )

    return q_stats


MyR2D2TorchPolicy = R2D2TorchPolicy.with_updates(
    optimizer_fn=optimizers.sgd_optimizer_dqn,
    loss_fn=my_r2d2_loss_wt_huber_loss,
    stats_fn=log.augment_stats_fn_wt_additionnal_logs(my_build_q_stats),
    before_init=policy.my_setup_early_mixins,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        policy.MyLearningRateSchedule,
    ],
)

MyR2D2TorchPolicyWtMSELoss = MyR2D2TorchPolicy.with_updates(
    loss_fn=my_r2d2_loss_wtout_huber_loss,
)

MyAdamR2D2TorchPolicy = MyR2D2TorchPolicy.with_updates(
    optimizer_fn=optimizers.adam_optimizer_dqn,
)

MyAdamVanillaR2D2TorchPolicy = R2D2TorchPolicy.with_updates(
    optimizer_fn=optimizers.adam_optimizer_dqn,
)
