from typing import Dict, Callable

import gym

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID, TensorType


def log_action_prob_pytorch(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """
    Log the mean of the probability of each actions, over the training batch.
    Also log the probabilities of the last step.
    Works only with the torch framework
    """
    # TODO make it work for other space than Discrete
    # TODO make is work for nested spaces
    to_log = {}
    if isinstance(policy.action_space, gym.spaces.Discrete):

        # DO not support nested discrete spaces
        assert train_batch["action_dist_inputs"].dim() == 2

        action_dist_inputs_avg = train_batch["action_dist_inputs"].mean(axis=0)
        action_dist_inputs_single = train_batch["action_dist_inputs"][-1, :]

        for action_i in range(policy.action_space.n):
            to_log[f"action_dist_inputs_avg"] = action_dist_inputs_avg[action_i]
            to_log[f"action_dist_inputs_single"] = action_dist_inputs_single[action_i]

        assert train_batch["action_prob"].dim() == 1
        to_log[f"action_prob_avg"] = train_batch["action_prob"].mean(axis=0)
        to_log[f"action_prob_single"] = train_batch["action_prob"][-1]
    else:
        raise NotImplementedError()
    return to_log

def log_env_info(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    to_log = {}

    # TODO is there a way without touching to the interceptor?
    get_interceptor = train_batch.get_interceptor
    train_batch.get_interceptor = None

    # This will only log the most recent infos (overwriting the older)
    # This may log unwanted stuff (and thus crash)
    # TODO Improve this like only logging train_batch['infos']['to_log']
    # TODO break after found one?
    # TODO Check for non TensorType
    for info in train_batch['infos'].tolist():
        to_log.update(info)

    train_batch.get_interceptor = get_interceptor

    return to_log


def stats_fn_wt_additionnal_logs(stats_function: Callable[[Policy, SampleBatch], Dict[str, TensorType]]):
    """
    Return a function executing the given function and adding additional logs about the TRAINING BATCH
    (not the actual actions)

    :param stats_function: the base stats function to use (args: [Policy, SampleBatch])
    :return: a function executing the stats_function and then adding additional logs
    """

    def wt_additional_info(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
        to_log = stats_function(policy, train_batch)

        # Additional logs
        # Log action probabilities
        if policy.config["framework"] == "torch":
            to_log.update(log_action_prob_pytorch(policy, train_batch))

        # Log info from custom env
        to_log.update(log_env_info(policy, train_batch))

        return to_log

    return wt_additional_info


def _get_to_log_from_policy(policy: Policy, policy_id: PolicyID) -> dict:
    """Gets the to_log var from a policy and rename its keys, adding the policy_id as a prefix."""
    to_log = {}
    if hasattr(policy, "to_log"):
        for k, v in policy.to_log.items():
            to_log[f"{policy_id}_{k}"] = v
        policy.to_log = {}
    return to_log


def update_train_result_wt_to_log(trainer, result: dict):
    """
    Add logs from every policies (from policy.to_log:dict) to the results (which are then plotted in Tensorboard).
    To be called from the on_train_result callback.
    """
    to_log_list = trainer.workers.foreach_policy(_get_to_log_from_policy)
    for i, to_log in enumerate(to_log_list):
        for k, v in to_log.items():
            key = f"worker_{i}_{k}"
            if key not in result.keys():
                result[key] = v
            else:
                raise ValueError(f"{key} already exists in result {result.keys()}")
