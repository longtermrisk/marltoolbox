import copy

import inspect
import time
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict


def sequence_of_fn_wt_same_args(*args, function_list, **kwargs) -> None:
    for fn in function_list:
        fn(*args, **kwargs)


def overwrite_config(dict_: dict, key, value):
    # assert isinstance(dict_, dict)
    # current_value = dict_
    # found = True
    # for k in key.split("."):
    #     if not found:
    #         print(f'Intermediary key: {k} not found in full key: {key}')
    #         return
    #     dict_ = current_value
    #     if k in current_value.keys():
    #         current_value = current_value[k]
    #     else:
    #         found = False
    dict_, k, current_value, found = move_to_key(dict_, key)

    if current_value != value:
        if found:
            print(f'Overwriting (key, value): ({key},{current_value}) with value: {value}')
            dict_[k] = value
        else:
            print(f'Adding (key, value): ({key},{value}) in dict.keys: {dict_.keys()}')
            dict_[k] = value


def move_to_key(dict_, key):
    # TODO make this more simple
    assert isinstance(dict_, dict)
    current_value = dict_
    found = True
    for k in key.split("."):
        if not found:
            print(f'Intermediary key: {k} not found in full key: {key}')
            return
        dict_ = current_value
        if k in current_value.keys():
            current_value = current_value[k]
        else:
            found = False
    return dict_, k, current_value, found


def extract_checkpoints(tune_experiment_analysis):
    all_best_checkpoints_per_trial = [
        tune_experiment_analysis.get_best_checkpoint(trial,
                                                     metric=tune_experiment_analysis.default_metric,
                                                     mode=tune_experiment_analysis.default_mode)
        for trial in tune_experiment_analysis.trials
    ]
    return all_best_checkpoints_per_trial


def extract_config_value(tune_experiment_analysis, key):
    values = []
    for trial in tune_experiment_analysis.trials:
        dict_, k, current_value, found = move_to_key(trial.config, key)
        if found:
            values.append(current_value)
        else:
            values.append(None)
    return values


def merge_callbacks(*callbacks_list):
    """
    Merge several callback class together. Executing them in the order provided.
    :param callbacks_list:
    :return: a class which calls all provided callbacks in order
    """

    callbacks_list = [callback() if inspect.isclass(callback) else callback for callback in callbacks_list]

    class MergeCallBacks(DefaultCallbacks):
        def __getattribute__(self, name):
            super_attr = super().__getattribute__(name)
            # Replace every callable by a callable calling the sequence of callbacks
            if callable(super_attr):
                def newfunc(*args, **kwargs):
                    for callbacks in callbacks_list:
                        function = callbacks.__getattribute__(name)
                        function(*args, **kwargs)

                return newfunc
            else:
                return super_attr

    return MergeCallBacks



def merge_policy_postprocessing_fn(*postprocessing_fn_list):
    """
    Merge several callback class together. Executing them in the order provided.
    :param postprocessing_fn_list:
    :return: a function which calls all provided function in order
    """
    def merged_postprocessing_fn(policy, sample_batch, other_agent_batches, episode):
        for postprocessing_fn in postprocessing_fn_list:
            sample_batch = postprocessing_fn (policy, sample_batch, other_agent_batches, episode)
        return sample_batch
    return merged_postprocessing_fn



def seed_to_checkpoint(dict_to_select_from):
    def get_value(policy_config):
        if "seed" in policy_config.keys():
            print("seed_to_checkpoint", policy_config["seed"])
            return dict_to_select_from[policy_config["seed"]]
        else:
            print('seed_to_checkpoint default to checkpoint 0. config["seed"]:', policy_config["seed"])
            return list(dict_to_select_from.values)[0]

    return get_value


def check_using_tune_class(config):
    return config.get("TuneTrainerClass", None) is not None


def set_config_for_evaluation(config: dict, policies_to_train=["None"]) -> dict:
    config_copy = copy.deepcopy(config)

    # Do not train
    # Always multiagent
    assert "multiagent" in config_copy.keys(), "Only working for config with multiagent key. " \
                                               f"config_copy.keys(): {config_copy.keys()}"
    config_copy["multiagent"]["policies_to_train"] = policies_to_train

    # Setup for evaluation
    # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    config_copy["explore"] = False

    # TODO below is really useless? If so then clean it
    # The following is not really needed since we are not training any policies
    # === Optimization ===
    # Learning rate for adam optimizer
    config_copy["lr"] = 0.0
    # # Learning rate schedule
    if "lr_schedule" in config_copy.keys():
        config_copy["lr_schedule"] = None

    return config_copy


def filter_tune_results(tune_analysis, metric, metric_threshold: float, metric_mode="last-5-avg",
                        threshold_mode="above"):
    assert threshold_mode in ("above", "equal", "below")
    assert metric_mode in ("avg", "min", "max", "last", "last-5-avg", "last-10-avg")
    print("Before trial filtering:", len(tune_analysis.trials), "trials")
    trials_filtered = []
    for trial in tune_analysis.trials:
        available_metrics = trial.metric_analysis
        if threshold_mode == "above" and available_metrics[metric][metric_mode] > metric_threshold:
            print("available_metrics[metric][metric_mode]", available_metrics[metric][metric_mode])
            print("metric_threshold", metric_threshold)
            trials_filtered.append(trial)
        elif threshold_mode == "equal" and available_metrics[metric][metric_mode] == metric_threshold:
            trials_filtered.append(trial)
        elif threshold_mode == "below" and available_metrics[metric][metric_mode] < metric_threshold:
            trials_filtered.append(trial)
    tune_analysis.trials = trials_filtered
    print("After trial filtering:", len(tune_analysis.trials), "trials")
    # TODO also filter the other stuff inside tune_analysis
    return tune_analysis


def get_random_seeds(n_seeds):
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(n_seeds))]
    return seeds


class PolicyCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        self._call_method_from_policies(worker, "on_episode_start")

    def on_episode_step(self, *, worker, base_env,
                        episode, env_index, **kwargs):
        self._call_method_from_policies(worker, "on_episode_step")

    def on_episode_end(self, *, worker, base_env,
                       policies, episode, env_index, **kwargs):
        self._call_method_from_policies(worker, "on_episode_end")

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs):
        self._call_method_from_policies(worker, "on_postprocess_trajectory")

    def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch,
                      **kwargs):
        self._call_method_from_policies(worker, "on_sample_end")

    def _call_method_from_policies(self, worker, method: str):
        for policy in worker.policy_map.values():
            if hasattr(policy, method) and callable(getattr(policy, method)):
                getattr(policy, method)()
