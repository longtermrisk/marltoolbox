from typing import Dict, Callable
import numbers
from collections import Iterable
import datetime
import os

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
        # print("train_batch", train_batch)
        # DO not support nested discrete spaces
        assert train_batch["action_dist_inputs"].dim() == 2

        action_dist_inputs_avg = train_batch["action_dist_inputs"].mean(axis=0)
        action_dist_inputs_single = train_batch["action_dist_inputs"][-1, :]

        for action_i in range(policy.action_space.n):
            to_log[f"act_dist_inputs_avg_{action_i}"] = action_dist_inputs_avg[action_i]
            to_log[f"act_dist_inputs_single_{action_i}"] = action_dist_inputs_single[action_i]

        assert train_batch["action_prob"].dim() == 1
        to_log[f"action_prob_avg"] = train_batch["action_prob"].mean(axis=0)
        to_log[f"action_prob_single"] = train_batch["action_prob"][-1]

        if "q_values" in train_batch.keys():
            assert train_batch["q_values"].dim() == 2
            q_values_avg = train_batch["q_values"].mean(axis=0)
            q_values_single = train_batch["q_values"][-1, :]

            for action_i in range(policy.action_space.n):
                to_log[f"q_values_avg_{action_i}"] = q_values_avg[action_i]
                to_log[f"q_values_single_{action_i}"] = q_values_single[action_i]


    else:
        raise NotImplementedError()
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
            to_log.update(_log_action_prob_pytorch(policy, train_batch))

        return to_log

    return wt_additional_info





#TODO problem this add max, min, avg of each log...
class LoggingCallbacks(DefaultCallbacks):
    VERBOSE = 1
    WEIGHTS_FREQUENCY = 10

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
        if self.VERBOSE > 0:
            self._add_env_info_to_custom_metrics(worker, episode)

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        if self.VERBOSE > 0:
            self._update_train_result_wt_to_log(trainer, result,
                                                function_to_exec=self._get_log_from_policy)
        if self.VERBOSE > 2:
            if not hasattr(self, "on_train_result_counter"):
                self.on_train_result_counter = 0
            if self.on_train_result_counter % self.WEIGHTS_FREQUENCY == 0:
                self._update_train_result_wt_to_log(trainer, result,
                                                    function_to_exec=self._get_weights_from_policy)
            self.on_train_result_counter += 1

    @staticmethod
    def _get_weights_from_policy(policy: Policy, policy_id: PolicyID) -> dict:
        """Gets the to_log var from a policy and rename its keys, adding the policy_id as a prefix."""
        to_log = {}
        weights = policy.get_weights()

        for k, v in weights.items():
            if isinstance(v, Iterable):
                to_log[f"{policy_id}/{k}"] = v

        return to_log

    @staticmethod
    def _add_env_info_to_custom_metrics(worker, episode):

        for agent_id in worker.policy_map.keys():
            info = episode.last_info_for(agent_id)
            for k, v in info.items():
                if isinstance(v, numbers.Number):
                    # TODO this add stuff as metrics (with mean, min, max) => available to select checkpoint but
                    #  produce a lot of logs !! Should be better if
                    episode.custom_metrics[f"{k}/{agent_id}"] = v


    def _update_train_result_wt_to_log(self, trainer, result: dict, function_to_exec):
        """
        Add logs from every policies (from policy.to_log:dict) to the results (which are then plotted in Tensorboard).
        To be called from the on_train_result callback.
        """
        def to_exec(worker):
            return worker.foreach_policy(function_to_exec)

        # to_log_list = trainer.workers.foreach_policy(function_to_exec)
        to_log_list_list = trainer.workers.foreach_worker(to_exec)
        for worker_idx, to_log_list in enumerate(to_log_list_list):
            for to_log in to_log_list:
                for k, v in to_log.items():
                    key = f"{k}/worker_{worker_idx}"
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
                to_log[f"{k}/{policy_id}"] = v
            policy.to_log = {}
        return to_log



def exp_name(name:str, n_days_worth_of_exp_in_same_dir=0,
             base_dir="~/ray_results", create_symlinks=True) -> str:
    """
    Give back f'{name}/YEAR_MONTH_DAY' and
    add symlinks to see the logs a few days old in this dir
    (to prevent long load time in tensorboard)
    :param name:
    :param n_days_worth_of_exp_in_same_dir:
    :param base_dir:
    :param create_symlinks:
    :return:
    """
    import datetime
    import os
    base_dir = os.path.expanduser(base_dir)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    intermediary_dir_path = os.path.join(base_dir, name)
    if not os.path.exists(intermediary_dir_path):
        os.mkdir(intermediary_dir_path)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    new_dir_name = os.path.join(name, date_str)
    new_dir_path = os.path.join(base_dir, new_dir_name)
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)

    if create_symlinks:
        past_dirs = []
        for i in range(n_days_worth_of_exp_in_same_dir):
            now = now - datetime.timedelta(1)
            date_str = now.strftime("%Y_%m_%d")
            past_dirs.append(os.path.join(name, date_str))

        for past_dir in past_dirs:
            path = os.path.join(base_dir, past_dir)
            if os.path.exists(path) and os.path.isdir(path):
                # link to child to prevent a chain of symlinks
                childs = [f.path for f in os.scandir(path) if f.is_dir()]
                for child_path in childs:
                    # child_path = os.path.join(path, child)
                    child_tail, child_head = os.path.split(child_path)
                    sym_link_path = os.path.join(new_dir_path, child_head)
                    if not os.path.exists(sym_link_path):
                        print("Create sym_link src:", child_path, "dst:", sym_link_path)
                        os.symlink(src=child_path, dst=sym_link_path)

    return new_dir_name


def put_everything_in_one_dir(exp_name):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    hour_str = now.strftime("%H_%M_%S")
    exp_name = os.path.join(exp_name,date_str)
    exp_name = os.path.join(exp_name,hour_str)
    base_dir = "~/ray_results"
    base_dir = os.path.expanduser(base_dir)
    exp_dir_path = os.path.join(base_dir, exp_name)
    return exp_name, exp_dir_path