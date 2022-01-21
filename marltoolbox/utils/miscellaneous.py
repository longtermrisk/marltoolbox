import copy
import difflib
import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch

if TYPE_CHECKING:
    pass

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
    # TODO use something similar by RLLib
    sub_struct, k, current_value, found = move_to_key(dict_, key)

    if current_value != value:
        if found:
            if (
                isinstance(current_value, tuple)
                and current_value[0] == OVERWRITE_KEY
            ):
                print(
                    f"NOT Overwriting (k: v): ({key}:{current_value}) "
                    f"with value: {value}.",
                    f"Instead overwriting with {current_value[1]} "
                    f"since OVERWRITE_KEY found",
                )
                sub_struct[k] = current_value[1]
            else:
                print(
                    f"Overwriting (k: v): ({key}:{current_value}) with "
                    f"value: {value}"
                )
                sub_struct[k] = value
        else:
            print(
                f"Adding (k: v): ({key}:{value}) in dict.keys:"
                f" {sub_struct.keys()}"
            )
            sub_struct[k] = value
    return dict_


def move_to_key(dict_: dict, key: str):
    """
    Get a value from nested dictionaries with '.' delimiting the keys.

    :param dict_: dict or nesyed dict
    :param key: key or serie of key joined by a '.'
    :return: Tuple(the lower level dict, lower level key, the final value,
        boolean for final value found)
    """
    assert isinstance(dict_, dict)
    current_value = dict_
    found = True
    for k in key.split("."):
        if not found:
            print(
                f"Intermediary key: {k} not found with full key: {key} "
                f"and dict: {dict_}"
            )
            return
        dict_ = current_value
        if k in current_value.keys():
            current_value = current_value[k]
        else:
            found = False
    return dict_, k, current_value, found


def merge_policy_postprocessing_fn(*postprocessing_fn_list):
    """
    Merge several callback class together.
    Executing them in the order provided.
    :param postprocessing_fn_list:
    :return: a function which calls all provided function in order
    """

    def merged_postprocessing_fn(
        policy, sample_batch, other_agent_batches, episode
    ):
        for postprocessing_fn in postprocessing_fn_list:
            sample_batch = postprocessing_fn(
                policy, sample_batch, other_agent_batches, episode
            )
        return sample_batch

    return merged_postprocessing_fn


def seed_to_checkpoint(dict_to_select_from: dict):
    def get_value(policy_config):
        if "seed" in policy_config.keys():
            print("seed_to_checkpoint", policy_config["seed"])
            return dict_to_select_from[policy_config["seed"]]
        else:
            print(
                "WARNING! seed_to_checkpoint default to checkpoint 0. "
                '"seed" not in policy_config.keys()'
            )
            return list(dict_to_select_from.values)[0]

    return get_value


def check_using_tune_class(config):
    return config.get("TuneTrainerClass", None) is not None


def set_config_for_evaluation(
    config: dict, policies_to_train=["None"]
) -> dict:
    config_copy = copy.deepcopy(config)

    # Always multiagent
    assert "multiagent" in config_copy.keys(), (
        "Only working for config with multiagent key. "
        f"config_copy.keys(): {config_copy.keys()}"
    )
    # Do not train
    config_copy["multiagent"]["policies_to_train"] = policies_to_train

    # Setup for evaluation
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


def get_random_seeds(n_seeds):
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(n_seeds))]
    return seeds


def ignore_str_containing_keys(str_list, ignore_keys):
    str_list_filtered = [
        file_path
        for file_path in str_list
        if all([key not in file_path for key in ignore_keys])
    ]
    print(
        len(str_list_filtered),
        "str remaining after ignoring str containing any ignore_keys:",
        ignore_keys,
    )
    return str_list_filtered


GROUP_KEY_NONE = "group_none"


def separate_str_in_group_containing_keys(str_list, group_keys):
    if len(group_keys) == 0:
        return {GROUP_KEY_NONE: str_list}

    groups_of_str_list = {}
    for group_key in group_keys:
        str_list_filtered = [
            file_path for file_path in str_list if group_key in file_path
        ]
        groups_of_str_list[f"group_{group_key}"] = str_list_filtered
        print(f"group {group_key} created with {len(str_list_filtered)} str")
    return groups_of_str_list


def keep_strs_containing_keys(str_list, plot_keys):
    str_list_filtered = [
        str_ for str_ in str_list if any([key in str_ for key in plot_keys])
    ]
    print(
        len(str_list_filtered),
        "str found after selecting plot_keys:",
        plot_keys,
    )
    return str_list_filtered


def fing_longer_substr(str_list):
    substr = ""
    if len(str_list) > 1 and len(str_list[0]) > 0:
        for i in range(len(str_list[0])):
            for j in range(len(str_list[0]) - i + 1):
                if j > len(substr) and all(
                    str_list[0][i : i + j] in x for x in str_list
                ):
                    substr = str_list[0][i : i + j]
    elif len(str_list) == 1:
        substr = str_list[0]
    return substr


def _get_experiment_state_file_path(one_checkpoint_path, split_path_n_times=1):
    one_checkpoint_path = os.path.expanduser(one_checkpoint_path)
    parent_dir = one_checkpoint_path
    for _ in range(split_path_n_times):
        parent_dir, head = os.path.split(parent_dir)
    json_file = "experiment_state-" + "_".join(head.split("_")[-2:]) + ".json"
    possible_files = os.listdir(parent_dir)
    json_file = difflib.get_close_matches(json_file, possible_files, n=1)[0]
    json_file_path = os.path.join(parent_dir, json_file)
    return json_file_path


def assert_if_key_in_dict_then_args_are_none(dict_, key, *args):
    if key in dict_.keys():
        for arg in args:
            assert arg is None


def read_from_dict_default_to_args(dict_, key, *args):
    if key in dict_.keys():
        return dict_[key]

    if len(args) == 1:
        return args[0]

    return args


def filter_sample_batch(
    samples: SampleBatch, filter_key, remove=True, copy_data=False
) -> SampleBatch:
    filter = samples.columns([filter_key])[0]
    if remove:
        assert isinstance(
            filter, np.ndarray
        ), f"type {type(filter)} for filter_key {filter_key}"
        filter = ~filter
    return SampleBatch(
        {k: np.array(v, copy=copy_data)[filter] for (k, v) in samples.items()}
    )
