import copy
import datetime
import logging
import math
import numbers
import os
import pickle
import pprint
import re
from collections import Iterable
from typing import Dict, Callable, TYPE_CHECKING, Optional

import gym
from scipy.special import softmax
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID, TensorType
from ray.util.debug import log_once
from marltoolbox.utils.log.full_epi_logger import FullEpisodeLogger
from marltoolbox.utils.log.model_summarizer import ModelSummarizer

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

logger = logging.getLogger(__name__)


def get_logging_callbacks_class(
    log_env_step: bool = True,
    log_from_policy: bool = True,
    log_full_epi: bool = False,
    log_full_epi_interval: int = 100,
    log_ful_epi_one_hot_obs: bool = True,
    log_weights: bool = False,
    log_weigths_interval: int = 100,
    add_worker_idx_to_key=False,
    log_model_sumamry: bool = True,
):
    class LoggingCallbacks(DefaultCallbacks):
        def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: MultiAgentEpisode,
            env_index: Optional[int] = None,
            **kwargs,
        ):
            if log_full_epi:
                if not self._is_full_episode_logging_initialized():
                    self._init_full_episode_logging(worker)
                self._full_episode_logger.on_episode_start()

            if log_model_sumamry:
                self._log_model_sumamry(worker)

        def _is_full_episode_logging_initialized(self):
            return hasattr(self, "_full_episode_logger")

        def _init_full_episode_logging(self, worker):
            self._full_episode_logger = FullEpisodeLogger(
                logdir=worker.io_context.log_dir,
                log_interval=log_full_epi_interval,
                log_ful_epi_one_hot_obs=log_ful_epi_one_hot_obs,
            )
            logger.info("_full_episode_logger init done")

        def _log_model_sumamry(self, worker):
            if log_once("model_summaries"):
                ModelSummarizer.for_every_policy_print_model_stats(worker)

        def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            episode: MultiAgentEpisode,
            env_index: Optional[int] = None,
            **kwargs,
        ):
            if log_env_step:
                self._add_env_info_to_custom_metrics(worker, episode)
            if log_full_epi:
                self._full_episode_logger.on_episode_step(episode)

        def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: MultiAgentEpisode,
            env_index: Optional[int] = None,
            **kwargs,
        ):
            if log_full_epi:
                self._full_episode_logger.on_episode_end(base_env)

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
                    trainer, result, function_to_exec=get_log_from_policy
                )
            if log_weights:
                if not hasattr(self, "on_train_result_counter"):
                    self.on_train_result_counter = 0
                if self.on_train_result_counter % log_weigths_interval == 0:
                    self._update_train_result_wt_to_log(
                        trainer,
                        result,
                        function_to_exec=self._get_weights_from_policy,
                    )
                self.on_train_result_counter += 1

        @staticmethod
        def _get_weights_from_policy(
            policy: Policy, policy_id: PolicyID
        ) -> dict:
            """Gets the to_log var from a policy and rename its keys, adding the policy_id as a prefix."""
            to_log = {}
            weights = policy.get_weights()

            for k, v in weights.items():
                if isinstance(v, Iterable):
                    to_log[f"{policy_id}/{k}"] = v

            return to_log

        @staticmethod
        def _add_env_info_to_custom_metrics(worker, episode):

            for policy_id in worker.policy_map.keys():
                info = episode.last_info_for(policy_id)
                for k, v in info.items():
                    if isinstance(v, numbers.Number):
                        # TODO this add the logs as metrics (with mean, min, max) => this does produce too much logs
                        episode.custom_metrics[f"{k}/{policy_id}"] = v

        def _update_train_result_wt_to_log(
            self, trainer, result: dict, function_to_exec
        ):
            """
            Add logs from every policies (from policy.to_log:dict)
            to the results (to be plotted in Tensorboard).
            To be called from the on_train_result callback.
            """

            def exec_in_each_policy(worker):
                return worker.foreach_policy(function_to_exec)

            # to_log_list = trainer.workers.foreach_policy(function_to_exec)
            to_log_list_list = trainer.workers.foreach_worker(
                exec_in_each_policy
            )
            for worker_idx, to_log_list in enumerate(to_log_list_list):
                for to_log in to_log_list:
                    for k, v in to_log.items():

                        if add_worker_idx_to_key:
                            key = f"{k}/worker_{worker_idx}"
                        else:
                            key = k

                        if key not in result.keys():
                            result[key] = v
                        else:
                            raise ValueError(
                                f"key:{key} already exists in result.keys(): {result.keys()}"
                            )

    return LoggingCallbacks


def get_log_from_policy(policy: Policy, policy_id: PolicyID) -> dict:
    """
    Gets the to_log var from a policy and rename its keys,
    adding the policy_id as a suffix.
    """
    to_log = {}
    if hasattr(policy, "to_log"):
        for k, v in policy.to_log.items():
            to_log[f"{k}/{policy_id}"] = v
        policy.to_log = {}
    return to_log


def augment_stats_fn_wt_additionnal_logs(
    stats_function: Callable[[Policy, SampleBatch], Dict[str, TensorType]]
):
    """
    Return a function executing the given function and adding additional logs about the TRAINING BATCH
    (not the actual actions)

    :param stats_function: the base stats function to use (args: [Policy, SampleBatch])
    :return: a function executing the stats_function and then adding additional logs
    """

    def wt_additional_info(
        policy: Policy, train_batch: SampleBatch
    ) -> Dict[str, TensorType]:
        stats_to_log = stats_function(policy, train_batch)

        # Additional logs
        # Log action probabilities
        if policy.config["framework"] == "torch":
            stats_to_log.update(_log_action_prob_pytorch(policy, train_batch))
        else:
            logger.warning(
                "wt_additional_info (stats_fn) working only for PyTorch"
            )

        return stats_to_log

    return wt_additional_info


def _log_action_prob_pytorch(
    policy: Policy, train_batch: SampleBatch
) -> Dict[str, TensorType]:
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
        if train_batch.ACTION_DIST_INPUTS in train_batch.keys():
            assert (
                train_batch[train_batch.ACTION_DIST_INPUTS].dim() == 2
            ), "Do not support nested discrete spaces"

            to_log = _add_action_distrib_to_log(policy, train_batch, to_log)
            to_log = _add_entropy_to_log(train_batch, to_log)
            to_log = _add_proba_of_action_played(train_batch, to_log)
            to_log = _add_q_values(policy, train_batch, to_log)
        else:
            logger.warning(
                "Key ACTION_DIST_INPUTS not found in train_batch. "
                "Can't perform _log_action_prob_pytorch."
            )
    else:
        raise NotImplementedError()
    return to_log


def _add_action_distrib_to_log(policy, train_batch, to_log):
    action_dist_inputs_avg = train_batch["action_dist_inputs"].mean(axis=0)
    action_dist_inputs_single = train_batch["action_dist_inputs"][-1, :]
    for action_i in range(policy.action_space.n):
        to_log[f"act_dist_inputs_avg_act{action_i}"] = action_dist_inputs_avg[
            action_i
        ]
        to_log[
            f"act_dist_inputs_single_act{action_i}"
        ] = action_dist_inputs_single[action_i]
        to_log[f"act_dist_inputs_single_max"] = max(action_dist_inputs_single)
    return to_log


def _add_entropy_to_log(train_batch, to_log):
    actions_proba_batch = train_batch["action_dist_inputs"]

    if _is_cuda_tensor(actions_proba_batch):
        actions_proba_batch = actions_proba_batch.cpu()

    if "q_values" in train_batch.keys():
        # Entropy of q_values used while playing in the environment
        # Theses q values has been transformed by the exploration
        actions_proba_batch = _convert_q_values_batch_to_proba_batch(
            actions_proba_batch
        )

    entropy_avg = _entropy_batch_proba_distrib(actions_proba_batch)
    entropy_single = _entropy_proba_distrib(actions_proba_batch[-1, :])
    to_log[f"entropy_buffer_samples_avg"] = entropy_avg
    to_log[f"entropy_buffer_samples_single"] = entropy_single

    return to_log


def _is_cuda_tensor(tensor):
    return hasattr(tensor, "is_cuda") and tensor.is_cuda


def _entropy_batch_proba_distrib(proba_distrib_batch):
    assert len(proba_distrib_batch) > 0
    entropy_batch = [
        _entropy_proba_distrib(proba_distrib_batch[batch_idx, ...])
        for batch_idx in range(len(proba_distrib_batch))
    ]
    mean_entropy = sum(entropy_batch) / len(entropy_batch)
    return mean_entropy


def _entropy_proba_distrib(proba_distrib):
    return sum([_entropy_proba(proba) for proba in proba_distrib])


def _entropy_proba(proba):
    assert proba >= 0.0, f"proba currently is {proba}"
    if proba == 0.0:
        return 0.0
    else:
        return -proba * math.log(proba)


def _add_proba_of_action_played(train_batch, to_log):
    assert train_batch["action_prob"].dim() == 1
    to_log[f"action_prob_avg"] = train_batch["action_prob"].mean(axis=0)
    to_log[f"action_prob_single"] = train_batch["action_prob"][-1]
    return to_log


def _convert_q_values_batch_to_proba_batch(q_values_batch):
    return softmax(q_values_batch, axis=1)


def _add_q_values(policy, train_batch, to_log):
    if "q_values" in train_batch.keys():
        assert train_batch["q_values"].dim() == 2
        q_values_avg = train_batch["q_values"].mean(axis=0)
        q_values_single = train_batch["q_values"][-1, :]
        for action_i in range(policy.action_space.n):
            to_log[f"q_values_avg_act{action_i}"] = q_values_avg[action_i]
            to_log[f"q_values_single_act{action_i}"] = q_values_single[
                action_i
            ]
            to_log[f"q_values_single_max"] = max(q_values_single)
    return to_log


def compute_entropy_from_raw_q_values(policy, q_values):
    actions_proba_batch = _apply_exploration(policy, dist_inputs=q_values)
    if _is_cuda_tensor(actions_proba_batch):
        actions_proba_batch = actions_proba_batch.cpu()
    actions_proba_batch = _convert_q_values_batch_to_proba_batch(
        actions_proba_batch
    )
    entropy_avg = _entropy_batch_proba_distrib(actions_proba_batch)
    entropy_single = _entropy_proba_distrib(actions_proba_batch[-1, :])
    return entropy_avg, entropy_single


def log_in_current_day_dir(exp_name):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    hour_str = now.strftime("%H_%M_%S")
    base_dir = os.getenv("TUNE_RESULT_DIR", "~/ray_results")
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
            {
                "metric_analysis": metric_analysis,
                "last_results": last_results,
                "config": config,
                "evaluated_params": evaluated_params,
            }
        )
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
        dict_or_list = [
            filter_nested(v, keywords_to_keep) for v in dict_or_list
        ]
        return dict_or_list

    dict_ = copy.deepcopy(dict_or_list)
    for k, v in dict_.items():
        if all(
            [re.search(keyword, k) is None for keyword in keywords_to_keep]
        ):
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


def log_learning_rate(policy):
    to_log = {}
    if hasattr(policy, "cur_lr"):
        to_log["cur_lr"] = policy.cur_lr
    for j, opt in enumerate(policy._optimizers):
        if hasattr(opt, "param_groups"):
            to_log[f"opt{j}_lr"] = [p["lr"] for p in opt.param_groups][0]
        else:
            print("opt doesn't have attr param_groups")
    return to_log


def _apply_exploration(policy, dist_inputs, explore=True):
    policy.exploration.before_compute_actions(
        explore=explore, timestep=policy.global_timestep
    )

    _, _ = policy.exploration.get_exploration_action(
        action_distribution=policy.dist_class(dist_inputs, policy.model),
        timestep=policy.global_timestep,
        explore=explore,
    )

    return dist_inputs
