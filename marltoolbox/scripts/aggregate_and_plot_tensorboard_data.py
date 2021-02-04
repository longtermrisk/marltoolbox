# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# Improved version of the original GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""
Aggregates multiple tensorboard runs into mean, min, max, median, std and save that in tensorboard files or in csv.
Also allows to plot the csv files created.

Usage example in a terminal:

The TensorBoard log files are in subfolders inside /LOLA_PG/2021_02_02/13_02_56 then:
python tensorboard_data_aggregate_and_plot.py
    --main_path /home/user/vm-data/ALGO_0/2021_02_02/13_02_56
    --plot_keys  [\'reward\'] --plot_aggregates  [\'mean\',\'min_max\']
    --plot_assemble_tags_in_one_plot [[\'reward_\']]

To aggregate and plot different experiments at the same time (but independently) you can use the group_keys.
group_keys (needs to be unique and in the path of each group):
python tensorboard_data_aggregate_and_plot.py
    --main_path /home/user/vm-data/
    --group_keys [\'09_18_50\',\'13_02_56\']
    --plot_keys  [\'reward\']
    --plot_aggregates  [\'mean\',\'min_max\']
    --plot_assemble_tags_in_one_plot [[\'reward_\']]
"""

import ast
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FOLDER_NAME = 'aggregates'
AGGREGATION_OPS = {"mean": np.mean, "min": np.min, "max": np.max, "median": np.median, "std": np.std, "var": np.var}
COLORS = list(mcolors.TABLEAU_COLORS)
GROUP_KEY_NONE = "group_none"

def extract_data(args):
    print("\n===== Extract data =====")
    file_list = list_all_files(args.main_path)
    file_list_filtered = ignore_files_using_keys(file_list, args.ignore_keys)
    file_list_dict = separate_files_in_group_using_keys(file_list_filtered, args.group_keys)

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    aggregate(args.main_path, args.output, file_list_dict)

def list_all_files(path):
    if not os.path.exists(path):
        raise FileExistsError(f"path doesn't exist: {path}")
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            file_list.append(os.path.join(root, file))
    print(len(file_list), "files found")
    return file_list

def ignore_files_using_keys(file_list, ignore_keys):
    file_list_filtered = [file_path for file_path in file_list if all([key not in file_path for key in ignore_keys])]
    print(len(file_list_filtered), "files remaining after ignoring files containing any ignore_keys:", ignore_keys)
    return file_list_filtered

def separate_files_in_group_using_keys(file_list, group_keys):
    if len(group_keys) == 0:
        return {GROUP_KEY_NONE: file_list}

    file_list_dict = {}
    for group_key in group_keys:
        file_list_filtered = [file_path for file_path in file_list if group_key in file_path]
        file_list_dict[f"group_{group_key}"] = file_list_filtered
        print(f"group {group_key} created with {len(file_list_filtered)} files")
    return file_list_dict

def aggregate(main_path, output, file_list_dict):

    print(f"Started aggregation {main_path}")

    extracts_per_group = {group_key: extract(main_path, file_list) for group_key, file_list in file_list_dict.items()}

    if output == "summary":
        raise NotImplementedError()
        # aggregate_to_summary(main_path, AGGREGATION_OPS, extracts_per_group)
    elif output == "csv":
        aggregate_to_csv(main_path, AGGREGATION_OPS, extracts_per_group)
    else:
        raise NotImplementedError(f"output format: {output} not implemented")

    print(f"Ended aggregation {main_path}")


# def aggregate_to_summary(dpath, aggregation_ops, extracts_per_group):
#     for key, aggregation_operation in aggregation_ops.items():
#         for group_key, all_per_key in extracts_per_group.items():
#             path = dpath / FOLDER_NAME / key / dpath.name / group_key
#             aggregations_per_key = {key: (steps, wall_times, aggregation_operation(values, axis=0)) for key, (steps, wall_times, values) in all_per_key.items()}
#             write_summary(path, aggregations_per_key)
#
#
# def write_summary(dpath, aggregations_per_key):
#     writer = tf.summary.FileWriter(dpath)
#
#     for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
#         for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
#             summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
#             scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
#             writer.add_event(scalar_event)
#
#         writer.flush()


def aggregate_to_csv(main_path, aggregation_ops, extracts_per_group):
    for group_key, all_per_key in extracts_per_group.items():
        for key, (steps, wall_times, values) in all_per_key.items():
            aggregations = {key: aggregation_operation(values, axis=0) for key, aggregation_operation in aggregation_ops.items()}
            write_csv(main_path, group_key, key, aggregations, steps)


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def write_csv(main_path, group_key, key, aggregations, steps):
    save_dir_path = os.path.join(main_path, FOLDER_NAME)
    main_path_split = os.path.split(main_path)

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    group_dir = get_valid_filename(group_key)
    save_group_dir = os.path.join(save_dir_path, group_dir) if group_key != GROUP_KEY_NONE else save_dir_path
    if not os.path.exists(save_group_dir):
        os.mkdir(save_group_dir)
    file_name = get_valid_filename(key) + '-' + main_path_split[-1] + '.csv'
    df = pd.DataFrame(aggregations, index=steps)
    save_dir_file_path = os.path.join(save_group_dir, file_name)
    df.to_csv(save_dir_file_path, sep=';')


def extract(main_path, file_list):
    print("Going to extract", main_path, "with len(file_list)", len(file_list))
    scalar_accumulators = [EventAccumulator(file_path).Reload(
        ).scalars for file_path in file_list]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]
    print(f"found {len(scalar_accumulators)} scalar_accumulators")

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]
    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    wall_times_per_key = [np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events], axis=0)
                          for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    keys = [key.replace("/", "_") for key in keys]
    all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))

    return all_per_key


def plot_tags(args):
    print("\n===== Plot =====")
    save_dir_path = os.path.join(args.main_path, FOLDER_NAME)
    file_list = list_all_files(save_dir_path)
    file_list = keep_files_using_keys(file_list, args.plot_keys)
    csv_file_list = [file_path for file_path in file_list if "csv" in file_path]
    csv_file_groups = separate_files_in_group_using_keys(csv_file_list, args.group_keys)

    for group_key, csv_files_in_one_group in csv_file_groups.items():
        save_dir_path_group = os.path.join(save_dir_path, group_key)
        if not os.path.exists(save_dir_path_group):
            os.mkdir(save_dir_path_group)

        print("===== Plot simple plots =====")
        for csv_file in csv_files_in_one_group:
            plot_several_tags(save_dir_path_group, [csv_file], args.plot_aggregates)

        print("===== Plot assemblages =====")
        plot_several_tags_per_plot(save_dir_path_group, csv_files_in_one_group,
                                   args.plot_assemble_tags_in_one_plot, args.plot_aggregates)

def keep_files_using_keys(file_list, plot_keys):
    file_list_filtered = [file_path for file_path in file_list if any([key in file_path for key in plot_keys])]
    print(len(file_list_filtered), "files found after selecting plot_keys:", plot_keys)
    return file_list_filtered

def plot_several_tags(save_dir_path, csv_file_list, plot_aggregates):
    fig = plt.figure()
    all_tags_seen = []
    for idx, csv_file in enumerate(csv_file_list):
        df = pd.read_csv(csv_file, sep=";")

        tag = extract_tag_from_file_name(csv_file)
        all_tags_seen.append(tag)

        metric_x = "logging_steps"
        metric_y = tag

        for col in df.columns:
            if col in plot_aggregates:
                plt.plot(df.index, df[col], label=f"{metric_y}: {col}", color=COLORS[idx])
        if "min_max" in plot_aggregates:
            plt.fill_between(df.index, df['min'], df['max'], color=COLORS[idx], alpha=0.2)
        if "one_std" in plot_aggregates:
            plt.fill_between(df.index, df['mean'] - df['std'], df['mean'] + df['std'], color=COLORS[idx], alpha=0.3)\

    plt.legend(numpoints=1, frameon=True)
    plt.xlabel(metric_x)
    metric_y = fing_longer_substr(all_tags_seen)
    plt.ylabel(metric_y)
    file_name = f'plot_{metric_y}_vs_{metric_x}.png'
    file_name = file_name.replace('/', '_')
    file_path = os.path.join(save_dir_path, file_name)
    print("save fig to", file_path)
    fig.savefig(file_path, dpi=fig.dpi)

def plot_several_tags_per_plot(save_dir_path, csv_file_list, plot_assemble_tags_in_one_plot, plot_aggregates):

    for list_of_tags_in_assemblage in plot_assemble_tags_in_one_plot:
        # select files for one assemblage
        assemblage_list = []
        for csv_file in csv_file_list:
            if any([select_key in csv_file for select_key in list_of_tags_in_assemblage]):
                assemblage_list.append(csv_file)
        # plot one assemblage
        plot_several_tags(save_dir_path, assemblage_list, plot_aggregates)

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


def extract_tag_from_file_name(csv_file):
    filename = os.path.split(csv_file)[-1]
    path_split = filename.split("-")[0]
    path_split = path_split.split("_")
    if path_split[0] == "ray":
        path_split = path_split[1:]
    if path_split[0] == "tune":
        path_split = path_split[1:]
    tag = "_".join(path_split)
    return tag




def main(args):
    print("args:", args)
    extract_data(args)

    if args.output == "csv":
        plot_tags(args)

def param_list(param):
    p_list = ast.literal_eval(param)
    if type(p_list) is not list:
        raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
    return p_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Args for aggregation data
    parser.add_argument("--main_path", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--group_keys", type=param_list, help="keys used to separate files in groups", default=[])
    parser.add_argument("--ignore_keys", type=param_list, help="keys used to ignore files",
                        default=["aggregates", "same_cross_play"])
    parser.add_argument("--output", type=str,
                        help="aggregation can be saves as tensorboard file (summary) or as table (csv)", default='csv')
    # Args for plotting
    parser.add_argument("--plot_keys", type=param_list,
                        help="keys used to select tensorboard tags to plot", default=['reward'])
    parser.add_argument("--plot_aggregates", type=param_list,
                        help="which results of aggregation operations to plot", default=['mean', 'min_max'])
    parser.add_argument("--plot_assemble_tags_in_one_plot", type=param_list,
                        help="keys used to select tensorboard tags to plot", default=[['reward']])

    args = parser.parse_args()
    main(args)
