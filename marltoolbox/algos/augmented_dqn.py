from typing import Dict

from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.dqn.dqn_torch_policy import compute_q_values, QLoss
from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_torch_policy import ComputeTDErrorMixin

import torch
import torch.nn.functional as F

from marltoolbox.utils import log, optimizers, policy


def build_q_losses_wt_additional_logs(policy: Policy, model, _,
                   train_batch: SampleBatch) -> TensorType:
    """
    Copy of build_q_losses with additional values saved into the policy
    Made only 2 changes, see in comments.
    """
    config = policy.config
    # Q-network evaluation.
    q_t, q_logits_t, q_probs_t = compute_q_values(
        policy,
        policy.q_model,
        train_batch[SampleBatch.CUR_OBS],
        explore=False,
        is_training=True)

    # Addition 1 out of 2
    policy.last_q_t = q_t.clone()

    # Target Q-network evaluation.
    q_tp1, q_logits_tp1, q_probs_tp1 = compute_q_values(
        policy,
        policy.target_q_model,
        train_batch[SampleBatch.NEXT_OBS],
        explore=False,
        is_training=True)

    # Addition 2 out of 2
    policy.last_target_q_t = q_tp1.clone()

    # Q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS],
                                  policy.action_space.n)
    q_t_selected = torch.sum(
        torch.where(q_t > FLOAT_MIN, q_t,
                    torch.tensor(0.0, device=policy.device)) *
        one_hot_selection, 1)
    q_logits_t_selected = torch.sum(
        q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1)

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        q_tp1_using_online_net, q_logits_tp1_using_online_net, \
            q_dist_tp1_using_online_net = compute_q_values(
                policy,
                policy.q_model,
                train_batch[SampleBatch.NEXT_OBS],
                explore=False,
                is_training=True)
        q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = F.one_hot(q_tp1_best_using_online_net,
                                                 policy.action_space.n)
        q_tp1_best = torch.sum(
            torch.where(q_tp1 > FLOAT_MIN, q_tp1,
                        torch.tensor(0.0, device=policy.device)) *
            q_tp1_best_one_hot_selection, 1)
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1)
    else:
        q_tp1_best_one_hot_selection = F.one_hot(
            torch.argmax(q_tp1, 1), policy.action_space.n)
        q_tp1_best = torch.sum(
            torch.where(q_tp1 > FLOAT_MIN, q_tp1,
                        torch.tensor(0.0, device=policy.device)) *
            q_tp1_best_one_hot_selection, 1)
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1)

    if PRIO_WEIGHTS not in train_batch.keys():
        assert config["prioritized_replay"] is False
        prio_weights = torch.tensor(
            [1.0] * len(train_batch[SampleBatch.REWARDS])).to(policy.device)
    else:
        prio_weights = train_batch[PRIO_WEIGHTS]


    policy.q_loss = QLoss(
        q_t_selected, q_logits_t_selected, q_tp1_best, q_probs_tp1_best,
        prio_weights, train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.DONES].float(), config["gamma"],
        config["n_step"], config["num_atoms"], config["v_min"],
        config["v_max"])

    return policy.q_loss.loss

def build_q_stats_wt_addtional_log(policy: Policy, batch) -> Dict[str,
                                                               TensorType]:
    entropy_avg, entropy_single = \
        log._compute_entropy_from_raw_q_values(
            policy, policy.last_q_t.clone())
    # entropy_target_avg, entropy_target_single = \
    #     log._compute_entropy_from_raw_q_values(
    #         policy, policy.last_target_q_t.clone())

    return dict({
        # "last_training_q_values": policy.last_q_t,
        # "last_training_target_q_values": policy.last_target_q_t,
        # "last_training_max_q_values": policy.last_q_t.max(),
        # "last_training_target_max_q_values": policy.last_target_q_t.max(),
        # "last_training_min_q_values": policy.last_q_t.min(),
        # "last_training_target_min_q_values": policy.last_target_q_t.min(),
        "entropy_avg": entropy_avg,
        # "entropy_single": entropy_single,
        # "entropy_target_avg": entropy_target_avg,
        # "entropy_target_single": entropy_target_single,
        "cur_lr": policy.cur_lr,
    }, **policy.q_loss.stats)


MyDQNTorchPolicy = DQNTorchPolicy.with_updates(
    optimizer_fn=optimizers.sgd_optimizer_dqn,
    loss_fn=build_q_losses_wt_additional_logs,
    stats_fn=log.augment_stats_fn_wt_additionnal_logs(
        build_q_stats_wt_addtional_log),
    before_init=policy.my_setup_early_mixins,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        policy.MyLearningRateSchedule,
    ]
)

MyAdamDQNTorchPolicy = MyDQNTorchPolicy.with_updates(
    optimizer_fn=optimizers.adam_optimizer_dqn,
)

