from typing import Dict, Callable

import gym

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType


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
        if policy.action_space.n <= 2:
            # Only one choice
            assert train_batch["action_prob"].dim() == 1
            to_log[f"action_prob_avg"] = train_batch["action_prob"].mean(axis=0)
            to_log[f"action_prob_single"] = train_batch["action_prob"][-1]
        else:
            assert train_batch["action_prob"].dim() == 2
            for i in range(policy.action_space.n):
                to_log[f"action_prob_{i}_avg"] = train_batch["action_prob"].mean(axis=0)[i]
                to_log[f"action_prob_{i}_single"] = train_batch["action_prob"][-1, i]

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

def log_policy_info(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    to_log = {}

    # TODO Check for non TensorType
    # TODO This can only get data from the Policy used for training (not the ones from the rollout workers),
    #  this create confusion (the given Policy is the one where learn_on_batch is call)
    if hasattr(policy, "to_log"):
        to_log.update(policy.to_log)
        policy.to_log = {}

    return to_log


def stats_fn_wt_additionnal_logs(stats_function: Callable[[Policy, SampleBatch], Dict[str, TensorType]]):
    """
    Return a function executing the given function and adding additional logs

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

        # Log info from custom policies
        to_log.update(log_policy_info(policy, train_batch))

        return to_log

    return wt_additional_info