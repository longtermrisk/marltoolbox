import copy
from typing import TYPE_CHECKING
import logging

import inspect
import os
import time
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

logger = logging.getLogger(__name__)

OVERWRITE_KEY = "OVERWRITE_KEY:"

def sequence_of_fn_wt_same_args(*args, function_list, **kwargs) -> None:
    for fn in function_list:
        fn(*args, **kwargs)


def overwrite_config(dict_: dict, key, value):
    """
    Helper to overwrite configuration file (with nested dictionaries inside)
    :param dict_: dict to edit
    :param key: string of the key to edit like: "first_key.intermediary_key.final_key_to_edit"
    :param value: value to write to for the "final_key_to_edit" key
    :return: dict_ edited
    """
    sub_struct, k, current_value, found = move_to_key(dict_, key)

    if current_value != value:
        if found:
            if isinstance(current_value, tuple) and current_value[0] == OVERWRITE_KEY:
                print(f'NOT Overwriting (key, value): ({key},{current_value}) with value: {value}',
                      f"Instead overwriting with {current_value[1]} since OVERWRITE_KEY found")
                sub_struct[k] = current_value[1]
            else:
                print(f'Overwriting (key, value): ({key},{current_value}) with value: {value}')
                sub_struct[k] = value
        else:
            print(f'Adding (key, value): ({key},{value}) in dict.keys: {sub_struct.keys()}')
            sub_struct[k] = value
    return dict_


def move_to_key(dict_, key):
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


def extract_config_values_from_tune_analysis(tune_experiment_analysis, key):
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

    class MergedCallBacks(DefaultCallbacks):
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

    return MergedCallBacks


def merge_policy_postprocessing_fn(*postprocessing_fn_list):
    """
    Merge several callback class together. Executing them in the order provided.
    :param postprocessing_fn_list:
    :return: a function which calls all provided function in order
    """

    def merged_postprocessing_fn(policy, sample_batch, other_agent_batches, episode):
        for postprocessing_fn in postprocessing_fn_list:
            sample_batch = postprocessing_fn(policy, sample_batch, other_agent_batches, episode)
        return sample_batch

    return merged_postprocessing_fn


def seed_to_checkpoint(dict_to_select_from: dict):
    def get_value(policy_config):
        if "seed" in policy_config.keys():
            print("seed_to_checkpoint", policy_config["seed"])
            return dict_to_select_from[policy_config["seed"]]
        else:
            print('WARNING! seed_to_checkpoint default to checkpoint 0. "seed" not in policy_config.keys()')
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
    # if "explore" in config_copy.keys():
    #     if config["explore"] == "OverwriteTrue":
    #         config_copy["explore"] = True
    #     elif config_copy["explore"] == "OverwriteFalse":
    #         config_copy["explore"] = False
    #     else:
    #         config_copy["explore"] = False
    # else:
    #     config_copy["explore"] = False
    overwrite_config(dict_=config_copy, key="explore", value=False)

    # TODO below is really useless (since are not training anyway)? If so then clean it
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
    print("metric_threshold", metric_threshold, "threshold_mode", threshold_mode)
    for trial_idx, trial in enumerate(tune_analysis.trials):
        available_metrics = trial.metric_analysis
        print(f"trial_idx {trial_idx} "
              f"available_metrics[{metric}][{metric_mode}] "
              f"{available_metrics[metric][metric_mode]}")
        if threshold_mode == "above" and available_metrics[metric][metric_mode] > metric_threshold:
            trials_filtered.append(trial)
        elif threshold_mode == "equal" and available_metrics[metric][metric_mode] == metric_threshold:
            trials_filtered.append(trial)
        elif threshold_mode == "below" and available_metrics[metric][metric_mode] < metric_threshold:
            trials_filtered.append(trial)
        else:
            print(f"filter trial {trial_idx}")
    tune_analysis.trials = trials_filtered
    print("After trial filtering:", len(tune_analysis.trials), "trials")
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


def list_all_files_in_one_dir_tree(path):
    if not os.path.exists(path):
        raise FileExistsError(f"path doesn't exist: {path}")
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            file_list.append(os.path.join(root, file))
    print(len(file_list), "files found")
    return file_list


def ignore_str_containing_keys(str_list, ignore_keys):
    str_list_filtered = [file_path for file_path in str_list if all([key not in file_path for key in ignore_keys])]
    print(len(str_list_filtered), "str remaining after ignoring str containing any ignore_keys:", ignore_keys)
    return str_list_filtered


GROUP_KEY_NONE = "group_none"


def separate_str_in_group_containing_keys(str_list, group_keys):
    if len(group_keys) == 0:
        return {GROUP_KEY_NONE: str_list}

    groups_of_str_list = {}
    for group_key in group_keys:
        str_list_filtered = [file_path for file_path in str_list if group_key in file_path]
        groups_of_str_list[f"group_{group_key}"] = str_list_filtered
        print(f"group {group_key} created with {len(str_list_filtered)} str")
    return groups_of_str_list


def keep_strs_containing_keys(str_list, plot_keys):
    str_list_filtered = [str_ for str_ in str_list if any([key in str_ for key in plot_keys])]
    print(len(str_list_filtered), "str found after selecting plot_keys:", plot_keys)
    return str_list_filtered


def fing_longer_substr(str_list):
    substr = ''
    if len(str_list) > 1 and len(str_list[0]) > 0:
        for i in range(len(str_list[0])):
            for j in range(len(str_list[0]) - i + 1):
                if j > len(substr) and all(str_list[0][i:i + j] in x for x in str_list):
                    substr = str_list[0][i:i + j]
    elif len(str_list) == 1:
        substr = str_list[0]
    return substr


def load_one_tune_analysis(checkpoints_paths:list):
    """Helper to re-create a fake tune_analysis only containing the
    checkpoints provided."""
    from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
    from ray.tune.trial import Trial
    from ray.tune.checkpoint_manager import Checkpoint
    from ray.tune import register_trainable
    from ray.tune import Trainable

    register_trainable("fake trial", Trainable)
    trials = []
    for one_checkpoint_path in checkpoints_paths:
        one_trial = Trial(trainable_name="fake trial")
        one_trial.checkpoint_manager.newest_persistent_checkpoint = \
            Checkpoint(Checkpoint.PERSISTENT, value=one_checkpoint_path)
        trials.append(one_trial)

    json_file_path = _get_experiment_state_file_path(checkpoints_paths[0])
    one_tune_analysis = ExperimentAnalysis(
        experiment_checkpoint_path=json_file_path,
        trials=trials,
        default_mode="max",
        default_metric="episode_reward_mean")
    return one_tune_analysis

def _get_experiment_state_file_path(one_checkpoint_path):
    import difflib
    parent_dir, head = os.path.split(one_checkpoint_path)
    json_file = "experiment_state-" + "_".join(head.split("_")[-2:]) + ".json"
    possible_files = os.listdir(parent_dir)
    json_file = difflib.get_close_matches(json_file, possible_files, n=1)[0]
    json_file_path = os.path.join(parent_dir, json_file)
    return json_file_path


def check_learning_achieved(tune_results, metric="episode_reward_mean",
                            trial_idx=0,
                            max_: float = None,
                            min_: float = None,
                            equal_: float = None):
    assert max_ is not None or min_ is not None or equal_ is not None

    logger.info(f"trial {trial_idx} achieved "
                f"{tune_results.trials[trial_idx].last_result[metric]}"
                f" on metric {metric}")
    print(f"trial {trial_idx} achieved "
                f"{tune_results.trials[trial_idx].last_result[metric]}"
                f" on metric {metric}")
    if min_ is not None:
        assert tune_results.trials[trial_idx].last_result[metric] > min_
    if max_ is not None:
        assert tune_results.trials[trial_idx].last_result[metric] < max_
    if equal_ is not None:
        assert tune_results.trials[trial_idx].last_result[metric] == equal_
