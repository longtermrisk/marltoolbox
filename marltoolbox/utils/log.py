from typing import Dict, Callable
import numbers

import gym

from ray.rllib.utils.typing import PolicyID, TensorType
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode

def _log_action_prob_pytorch(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """
    Log the mean of the probability of each actions, over the training batch.
    Also log the probabilities of the last step.
    Works only with the torch framework
    """
    # TODO make it work for other space than Discrete
    # TODO make is work for nested spaces
    # TODO add entropy
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

# def log_env_info(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
#     to_log = {}
#
#     # TODO is there a way without touching to the interceptor?
#     get_interceptor = train_batch.get_interceptor
#     train_batch.get_interceptor = None
#
#     # This will only log the last infos in the bach (overwriting the other)
#     # This is unrealiable since the batch can be sampled from a buffer
#     # This may log unwanted stuff (and thus crash)
#     # TODO Improve this like only logging train_batch['infos']['to_log']
#     # TODO break after found one?
#     # TODO Check for non TensorType
#     for info in train_batch['infos'].tolist():
#         to_log.update(info)
#
#     train_batch.get_interceptor = get_interceptor
#
#     return to_log


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
            to_log.update(_log_action_prob_pytorch(policy, train_batch))

        return to_log

    return wt_additional_info






class LoggingCallbacks(DefaultCallbacks):

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        """Runs on each episode step.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (int): The index of the (vectorized) env, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """

        self._add_env_info_to_custom_metrics(worker, episode)

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        self._update_train_result_wt_to_log(trainer, result)

    @staticmethod
    def _add_env_info_to_custom_metrics(worker, episode):

        for agent_id in worker.policy_map.keys():
            info = episode.last_info_for(agent_id)
            for k, v in info.items():
                if isinstance(v, numbers.Number):
                    episode.custom_metrics[f"{k}_{agent_id}"] = v


    def _update_train_result_wt_to_log(self, trainer, result: dict):
        """
        Add logs from every policies (from policy.to_log:dict) to the results (which are then plotted in Tensorboard).
        To be called from the on_train_result callback.
        """
        to_log_list = trainer.workers.foreach_policy(self._get_log_from_policy)
        for i, to_log in enumerate(to_log_list):
            for k, v in to_log.items():
                key = f"worker_{i}_{k}"
                if key not in result.keys():
                    result[key] = v
                else:
                    raise ValueError(f"key:{key} already exists in result.keys(): {result.keys()}")


    @staticmethod
    def _get_log_from_policy(policy: Policy, policy_id: PolicyID) -> dict:
        """Gets the to_log var from a policy and rename its keys, adding the policy_id as a prefix."""
        to_log = {}
        if hasattr(policy, "to_log"):
            for k, v in policy.to_log.items():
                to_log[f"{k}_{policy_id}"] = v
            policy.to_log = {}
        return to_log
