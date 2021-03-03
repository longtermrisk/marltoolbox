import copy
import datetime
import numbers
import os
import pickle
import pprint
import re
from collections import Iterable
from typing import Dict, Callable, TYPE_CHECKING

import gym
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID, TensorType

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

def get_logging_callbacks_class(log_env_step=True,
                                log_from_policy=True,
                                log_full_epi=False, log_full_epi_delay=100,
                                log_weights=False, log_weigths_interval=100):
    class LoggingCallbacks(DefaultCallbacks):

        def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                            episode: MultiAgentEpisode, env_index: int,
                            **kwargs):
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
            if log_env_step:
                self._add_env_info_to_custom_metrics(worker, episode)
            if log_full_epi and self.log_full_epi_steps:
                self._add_step_in_buffer(episode)

        def _add_step_in_buffer(self, episode):
            data = {}
            for agent_id in episode._policies.keys():
                action = episode.last_action_for(agent_id).tolist()
                info = episode.last_info_for(agent_id)
                epi = episode.episode_id
                # TODO make this clean
                if isinstance(action, Iterable) and len(action) == 1:
                    action = action[0]
                rewards = episode._agent_reward_history[agent_id]
                reward = rewards[-1].tolist() if len(rewards) > 0 else None
                if reward is not None and isinstance(reward, Iterable) and len(
                        reward) == 1:
                    reward = reward[0]
                data[agent_id] = {"action": action, "reward": reward,
                                  "info": info, "epi": epi}
            self.buffer.append(data)

        def on_episode_start(self, *, worker: "RolloutWorker",
                             base_env: BaseEnv,
                             policies: Dict[PolicyID, Policy],
                             episode: MultiAgentEpisode, env_index: int,
                             **kwargs):
            """Callback run on the rollout worker before each episode starts.

            Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                base_env (BaseEnv): BaseEnv running the episode. The underlying
                    env object can be gotten by calling base_env.get_unwrapped().
                policies (dict): Mapping of policy id to policy objects. In single
                    agent mode there will only be a single "default" policy.
                episode (MultiAgentEpisode): Episode object which contains episode
                    state. You can use the `episode.user_data` dict to store
                    temporary data, and `episode.custom_metrics` to store custom
                    metrics for the episode.
                env_index (int): The index of the (vectorized) env, which the
                    episode belongs to.
                kwargs: Forward compatibility placeholder.
            """
            if log_full_epi:
                if not hasattr(self, "log_full_epi_step"):
                    self.log_full_epi_steps = True
                    self.buffer = []
                    self.delay_counter = log_full_epi_delay

                if self.buffer == []:
                    if self.delay_counter >= log_full_epi_delay:
                        self.log_full_epi_steps = True
                        self.delay_counter = 0

        def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                           policies: Dict[PolicyID, Policy],
                           episode: MultiAgentEpisode, env_index: int,
                           **kwargs):
            """Runs when an episode is done.

            Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                base_env (BaseEnv): BaseEnv running the episode. The underlying
                    env object can be gotten by calling base_env.get_unwrapped().
                policies (dict): Mapping of policy id to policy objects. In single
                    agent mode there will only be a single "default" policy.
                episode (MultiAgentEpisode): Episode object which contains episode
                    state. You can use the `episode.user_data` dict to store
                    temporary data, and `episode.custom_metrics` to store custom
                    metrics for the episode.
                env_index (int): The index of the (vectorized) env, which the
                    episode belongs to.
                kwargs: Forward compatibility placeholder.
            """
            if log_full_epi:
                self.log_full_epi_steps = False

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            """Called at the end of Trainable.train().

            Args:
                trainer (Trainer): Current trainer instance.
                result (dict): Dict of results returned from trainer.train() call.
                    You can mutate this object to add additional metrics.
                kwargs: Forward compatibility placeholder.
            """
            if log_from_policy:
                self._update_train_result_wt_to_log(
                    trainer, result,
                    function_to_exec=self._get_log_from_policy)
            if log_weights:
                if not hasattr(self, "on_train_result_counter"):
                    self.on_train_result_counter = 0
                if self.on_train_result_counter % log_weigths_interval == 0:
                    self._update_train_result_wt_to_log(
                        trainer, result,
                        function_to_exec=self._get_weights_from_policy)
                self.on_train_result_counter += 1

            if log_full_epi:
                if not hasattr(self, "delay_counter"):
                    self.delay_counter = 0
                self.delay_counter += 1
                self._get_log_from_buffer(trainer, result)

        @staticmethod
        def _get_weights_from_policy(policy: Policy,
                                     policy_id: PolicyID) -> dict:
            """Gets the to_log var from a policy and rename its keys, adding the policy_id as a prefix."""
            to_log = {}
            weights = policy.get_weights()

            for k, v in weights.items():
                if isinstance(v, Iterable):
                    to_log[f"{policy_id}/{k}"] = v

            return to_log

        def _get_log_from_buffer(self, trainer, result) -> dict:
            """Gets the to_log var from a policy and rename its keys, adding the policy_id as a prefix."""
            if len(self.buffer) > 0:
                data = self.buffer.pop(0)
                for agent_id, agent_data in data.items():
                    for k, v in agent_data.items():
                        key = f"intra_epi_{agent_id}_{k}"
                        if key not in result.keys():
                            result[key] = v
                        else:
                            raise ValueError(
                                f"key:{key} already exists in result.keys(): {result.keys()}")

        @staticmethod
        def _add_env_info_to_custom_metrics(worker, episode):

            for agent_id in worker.policy_map.keys():
                info = episode.last_info_for(agent_id)
                for k, v in info.items():
                    if isinstance(v, numbers.Number):
                        # TODO this add the logs as metrics (with mean, min, max) => this does produce too much logs
                        episode.custom_metrics[f"{k}/{agent_id}"] = v

        def _update_train_result_wt_to_log(self, trainer, result: dict,
                                           function_to_exec):
            """
            Add logs from every policies (from policy.to_log:dict)
            to the results (to be plotted in Tensorboard).
            To be called from the on_train_result callback.
            """

            def exec_in_each_policy(worker):
                return worker.foreach_policy(function_to_exec)

            # to_log_list = trainer.workers.foreach_policy(function_to_exec)
            to_log_list_list = trainer.workers.foreach_worker(
                exec_in_each_policy)
            for worker_idx, to_log_list in enumerate(to_log_list_list):
                for to_log in to_log_list:
                    for k, v in to_log.items():
                        key = f"{k}/worker_{worker_idx}"
                        if key not in result.keys():
                            result[key] = v
                        else:
                            raise ValueError(
                                f"key:{key} already exists in result.keys(): {result.keys()}")

        @staticmethod
        def _get_log_from_policy(policy: Policy, policy_id: PolicyID) -> dict:
            """Gets the to_log var from a policy and rename its keys, adding the policy_id as a prefix."""
            to_log = {}
            if hasattr(policy, "to_log"):
                for k, v in policy.to_log.items():
                    to_log[f"{k}/{policy_id}"] = v
                policy.to_log = {}
            return to_log

    return LoggingCallbacks


def _log_action_prob_pytorch(policy: Policy, train_batch: SampleBatch) -> Dict[
    str, TensorType]:
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
            to_log[f"act_dist_inputs_avg_{action_i}"] = action_dist_inputs_avg[
                action_i]
            to_log[f"act_dist_inputs_single_{action_i}"] = \
                action_dist_inputs_single[action_i]

        assert train_batch["action_prob"].dim() == 1
        to_log[f"action_prob_avg"] = train_batch["action_prob"].mean(axis=0)
        to_log[f"action_prob_single"] = train_batch["action_prob"][-1]

        if "q_values" in train_batch.keys():
            assert train_batch["q_values"].dim() == 2
            q_values_avg = train_batch["q_values"].mean(axis=0)
            q_values_single = train_batch["q_values"][-1, :]

            for action_i in range(policy.action_space.n):
                to_log[f"q_values_avg_{action_i}"] = q_values_avg[action_i]
                to_log[f"q_values_single_{action_i}"] = q_values_single[
                    action_i]


    else:
        raise NotImplementedError()
    return to_log


def stats_fn_wt_additionnal_logs(
        stats_function: Callable[[Policy, SampleBatch], Dict[str, TensorType]]):
    """
    Return a function executing the given function and adding additional logs about the TRAINING BATCH
    (not the actual actions)

    :param stats_function: the base stats function to use (args: [Policy, SampleBatch])
    :return: a function executing the stats_function and then adding additional logs
    """

    def wt_additional_info(policy: Policy, train_batch: SampleBatch) -> Dict[
        str, TensorType]:
        to_log = stats_function(policy, train_batch)

        # Additional logs
        # Log action probabilities
        if policy.config["framework"] == "torch":
            to_log.update(_log_action_prob_pytorch(policy, train_batch))

        return to_log

    return wt_additional_info


def log_in_current_day_dir(exp_name):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    hour_str = now.strftime("%H_%M_%S")
    base_dir = os.getenv('TUNE_RESULT_DIR', "~/ray_results")
    base_dir = os.path.expanduser(base_dir)
    exp_name = os.path.join(exp_name, date_str, hour_str)
    exp_dir_path = os.path.join(base_dir, exp_name)
    return exp_name, exp_dir_path


def extract_all_metrics_from_results(results, limit=False):
    metrics = []
    for trial in results.trials:
        metric_analysis = trial.metric_analysis
        config = trial.config
        if limit:
            last_results = None
            config.pop("callbacks", None)
            config.pop("multiagent", None)
        else:
            last_results = trial.last_result
        evaluated_params = trial.evaluated_params

        metrics.append(
            {"metric_analysis": metric_analysis, "last_results": last_results,
             "config": config, "evaluated_params": evaluated_params})
    return metrics


def save_metrics(results, exp_name, filename, limit=False):
    save_path = os.path.join(f"~/ray_results", exp_name, filename)
    save_path = os.path.expanduser(save_path)
    with open(save_path, "wb") as f:
        metrics = extract_all_metrics_from_results(results, limit=limit)
        print("metrics", metrics)
        pickle.dump(metrics, f)


def filter_nested(dict_or_list, keywords_to_keep):
    if isinstance(dict_or_list, list):
        dict_or_list = [filter_nested(v, keywords_to_keep) for v in
                        dict_or_list]
        return dict_or_list

    dict_ = copy.deepcopy(dict_or_list)
    for k, v in dict_.items():
        if all([re.search(keyword, k) is None for keyword in keywords_to_keep]):
            dict_or_list.pop(k)
        else:
            if isinstance(v, dict) or isinstance(v, list):
                dict_or_list[k] = filter_nested(v, keywords_to_keep)
    return dict_or_list


def pprint_saved_metrics(file_path, keywords_to_print=None):
    pp = pprint.PrettyPrinter(depth=4)
    with open(file_path, "rb") as f:
        metrics = pickle.load(f)
        if keywords_to_print is not None:
            metrics = filter_nested(metrics, keywords_to_print)
        pp.pprint(metrics)
