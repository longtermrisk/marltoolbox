import json
import os
from typing import List

from marltoolbox.utils import miscellaneous
from marltoolbox.utils.tune_analysis import ABOVE, BELOW, EQUAL


def get_unique_child_dir(_dir: str):
    """
    Return the path to the unique dir inside the given dir.

    :param _dir: path to given dir
    :return: path to the unique dir inside the given dir
    """

    list_child_dir = os.listdir(_dir)
    list_child_dir = [
        os.path.join(_dir, child_dir) for child_dir in list_child_dir
    ]
    list_child_dir = keep_dirs_only(list_child_dir)
    assert len(list_child_dir) == 1, f"{list_child_dir}"
    unique_child_dir = list_child_dir[0]
    return unique_child_dir


def try_get_unique_child_dir(_dir: str):
    """
    If it exists, returns the path to the unique dir inside the given dir.
    Otherwise returns None.

    :param _dir: path to given dir
    :return: path to the unique dir inside the given dir or if it doesn't
        exist None
    """

    try:
        unique_child_dir = get_unique_child_dir(_dir)
        return unique_child_dir
    except AssertionError:
        return None


def list_all_files_in_one_dir_tree(path: str) -> List[str]:
    """
    List all the files in the tree starting at the given path.

    :param path:
    :return: list of all the files
    """
    if not os.path.exists(path):
        raise FileExistsError(f"path doesn't exist: {path}")
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            file_list.append(os.path.join(root, file))
    print(len(file_list), "files found")
    return file_list


def get_children_paths_wt_selecting_filter(
    parent_dir_path: str, _filter: str
) -> List[str]:
    """
    Return all children dir paths after selecting those containing the
    _filter.

    :param parent_dir_path:
    :param _filter: to select the paths to keep
    :return: list of paths which contain the given filter.
    """
    return _get_children_paths_filters(
        parent_dir_path, selecting_filter=_filter
    )


def get_children_paths_wt_discarding_filter(
    parent_dir_path: str, _filter: str
) -> List[str]:
    """
    Return all children dir paths after selecting those NOT containing the
    _filter.

    :param parent_dir_path:
    :param _filter: to select the paths to remove
    :return: list of paths which don't contain the given filter.
    """

    return _get_children_paths_filters(
        parent_dir_path, discarding_filter=_filter
    )


def _get_children_paths_filters(
    parent_dir_path: str,
    selecting_filter: str = None,
    discarding_filter: str = None,
):
    filtered_children = os.listdir(parent_dir_path)
    if selecting_filter is not None:
        filtered_children = [
            filename
            for filename in filtered_children
            if selecting_filter in filename
        ]
    if discarding_filter is not None:
        filtered_children = [
            filename
            for filename in filtered_children
            if discarding_filter not in filename
        ]
    filtered_children_path = [
        os.path.join(parent_dir_path, filename)
        for filename in filtered_children
    ]
    return filtered_children_path


def get_params_for_replicate(trial_dir_path: str) -> dict:
    """
    Get the parameters from the json file saved in the dir of an Tune/RLLib
    trial.

    :param trial_dir_path: patht to a single tune.Trial (inside an experiment)
    :return: dict of parameters used for the trial
    """
    parameter_json_path = os.path.join(trial_dir_path, "params.json")
    params = _read_json_file(parameter_json_path)
    return params


def get_results_for_replicate(trial_dir_path: str) -> list:
    """
    Get the results for all episodes from the file saved in the
    dir of an Tune/RLLib trial.

    :param trial_dir_path: patht to a single tune.Trial (inside an experiment)
    :return: list of lines of results (one line per episode)
    """
    results_file_path = os.path.join(trial_dir_path, "result.json")
    results = _read_all_lines_of_file(results_file_path)
    # Remove empty last line
    if len(results[-1]) == 0:
        results = results[:-1]
    results = [json.loads(line) for line in results]
    return results


def _read_json_file(json_file_path: str):
    with open(json_file_path) as json_file:
        json_object = json.load(json_file)
    return json_object


def _read_all_lines_of_file(file_path: str) -> list:
    with open(file_path) as file:
        lines = list(file)
    return lines


def keep_dirs_only(paths: list) -> list:
    """Keep only the directories"""
    return [path for path in paths if os.path.isdir(path)]


def filter_list_of_replicates_by_results(
    replicate_paths: list,
    filter_key: str,
    filter_threshold: float,
    filter_mode: str = ABOVE,
) -> list:
    print("Going to start filtering replicate_paths")
    print("len(replicate_paths)", len(replicate_paths))
    filtered_replicate_paths = []
    for replica_path in replicate_paths:
        replica_results = get_results_for_replicate(replica_path)
        last_result = replica_results[-1]
        assert isinstance(last_result, dict)
        _, _, current_value, found = miscellaneous.move_to_key(
            last_result, filter_key
        )
        assert found, (
            f"filter_key {filter_key} not found in last_result "
            f"{last_result}"
        )
        if filter_mode == ABOVE and current_value > filter_threshold:
            filtered_replicate_paths.append(replica_path)
        elif filter_mode == EQUAL and current_value == filter_threshold:
            filtered_replicate_paths.append(replica_path)
        elif filter_mode == BELOW and current_value < filter_threshold:
            filtered_replicate_paths.append(replica_path)
        else:
            print(f"filtering out replica_path {replica_path}")
    print("After filtering:")
    print("len(filtered_replicate_paths)", len(filtered_replicate_paths))
    return filtered_replicate_paths
