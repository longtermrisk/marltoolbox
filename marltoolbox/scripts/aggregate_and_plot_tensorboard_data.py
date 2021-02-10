# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# Improved version of the original GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""
Aggregates multiple tensorboard runs into mean, min, max, median, std and save that in tensorboard files or in csv.
Also allows to plot the csv files created.

Usage example in a terminal:

The TensorBoard log files are in subfolders inside /LOLA_PG/2021_02_02/13_02_56 then:
python aggregate_and_plot_tensorboard_data.py
    --main_path /home/user/vm-data/ALGO_0/2021_02_02/13_02_56
    --plot_keys  [\'reward\'] --plot_aggregates  [\'mean\',\'min_max\']
    --plot_assemble_tags_in_one_plot [[\'reward_\']]

To aggregate and plot different experiments at the same time (but independently) you can use the group_keys.
group_keys (needs to be unique and in the path of each group):
python aggregate_and_plot_tensorboard_data.py
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
from marltoolbox.utils.plot import LOWER_ENVELOPE_SUFFIX, UPPER_ENVELOPE_SUFFIX, PlotHelper, PlotConfig
from marltoolbox.utils.miscellaneous import list_all_files_in_one_dir_tree, ignore_str_containing_keys, \
    separate_str_in_group_containing_keys, GROUP_KEY_NONE, keep_strs_containing_keys, fing_longer_substr
FOLDER_NAME = 'aggregates'
AGGREGATION_OPS = {"mean": np.mean, "min": np.min, "max": np.max, "median": np.median, "std": np.std, "var": np.var}
COLORS = list(mcolors.TABLEAU_COLORS)




class TensorBoardDataExtractor():

    def extract_data(self, main_path, ignore_keys, group_keys, output):
        print("\n===== Extract data =====")
        file_list = list_all_files_in_one_dir_tree(main_path)
        file_list_filtered = ignore_str_containing_keys(file_list, ignore_keys)
        file_list_dict = separate_str_in_group_containing_keys(file_list_filtered, group_keys)

        if output not in ['summary', 'csv']:
            raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(output))

        self._aggregate(main_path, output, file_list_dict)


    def _aggregate(self,main_path, output, file_list_dict):

        print(f"Started aggregation {main_path}")

        extracts_per_group = {group_key: self._extract_x_y_per_keys(main_path, file_list) for group_key, file_list in
                              file_list_dict.items()}

        if output == "summary":
            raise NotImplementedError()
            # Look at the original implementation at https://github.com/Spenhouet/tensorboard-aggregator
            # aggregate_to_summary(main_path, AGGREGATION_OPS, extracts_per_group)
        elif output == "csv":
            self._aggregate_to_csv(main_path, AGGREGATION_OPS, extracts_per_group)
        else:
            raise NotImplementedError(f"output format: {output} not implemented")

        print(f"Ended aggregation {main_path}")

    def _create_event_reader_for_each_log_files(self, file_list):
        event_readers = [EventAccumulator(file_path).Reload(
        ).scalars for file_path in file_list]

        # Filter non event files
        event_readers = [one_event_reader for one_event_reader in event_readers if
                               one_event_reader.Keys()]
        print(f"found {len(event_readers)} event_readers")
        return event_readers

    def _get_and_validate_all_scalar_keys(self, event_readers):

        all_keys = [tuple(one_event_reader.Keys()) for one_event_reader in event_readers]
        assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(
            all_keys)
        keys = all_keys[0]

        all_scalar_events_per_key = [[one_event_reader.Items(key) for one_event_reader in event_readers] for
                                     key in keys]
        return all_scalar_events_per_key, keys

    def _get_and_validate_all_steps_per_key(self, all_scalar_events_per_key, keys):
        all_steps_per_key = [
            [tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
            for all_scalar_events in all_scalar_events_per_key]

        for i, all_steps in enumerate(all_steps_per_key):
            assert len(set(
                all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
                keys[i], [len(steps) for steps in all_steps])

        steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

        return steps_per_key

    def _get_values_per_step_per_key(self, all_scalar_events_per_key):
        values_per_key = [
            [[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
            for all_scalar_events in all_scalar_events_per_key]
        return values_per_key

    def _extract_x_y_per_keys(self, main_path, file_list):
        print("Going to extract", main_path, "with len(file_list)", len(file_list))

        event_readers = self._create_event_reader_for_each_log_files(file_list)
        all_scalar_events_per_key, keys = self._get_and_validate_all_scalar_keys(event_readers)
        steps_per_key = self._get_and_validate_all_steps_per_key(all_scalar_events_per_key, keys)
        values_per_key = self._get_values_per_step_per_key(all_scalar_events_per_key)

        keys = [key.replace("/", "_") for key in keys]
        all_per_key = dict(zip(keys, zip(steps_per_key, values_per_key)))

        return all_per_key


    def _aggregate_to_csv(self,main_path, aggregation_ops, extracts_per_group):
        for group_key, all_per_key in extracts_per_group.items():
            for key, (steps, values) in all_per_key.items():
                aggregations = {key: aggregation_operation(values, axis=0) for key, aggregation_operation in aggregation_ops.items()}
                self._write_csv(main_path, group_key, key, aggregations, steps)

    def _write_csv(self,main_path, group_key, key, aggregations, steps):
        save_dir_path = os.path.join(main_path, FOLDER_NAME)
        main_path_split = os.path.split(main_path)

        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        group_dir = self._get_valid_filename(group_key)
        save_group_dir = os.path.join(save_dir_path, group_dir) if group_key != GROUP_KEY_NONE else save_dir_path
        if not os.path.exists(save_group_dir):
            os.mkdir(save_group_dir)
        file_name = self._get_valid_filename(key) + '-' + main_path_split[-1] + '.csv'
        df = pd.DataFrame(aggregations, index=steps)
        save_dir_file_path = os.path.join(save_group_dir, file_name)
        df.to_csv(save_dir_file_path, sep=';')

    def _get_valid_filename(self,s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)




def plot_selected_keys(args):
    print("\n===== Plot =====")
    save_dir_path = os.path.join(args.main_path, FOLDER_NAME)
    file_list = list_all_files_in_one_dir_tree(save_dir_path)
    file_list = keep_strs_containing_keys(file_list, args.plot_keys)
    csv_file_list = [file_path for file_path in file_list if "csv" in file_path]
    csv_file_groups = separate_str_in_group_containing_keys(csv_file_list, args.group_keys)

    for group_key, csv_files_in_one_group in csv_file_groups.items():
        save_dir_path_group = os.path.join(save_dir_path, group_key)
        if not os.path.exists(save_dir_path_group):
            os.mkdir(save_dir_path_group)

        print("===== Plot simple plots =====")
        for csv_file in csv_files_in_one_group:
            plot_one_graph(save_dir_path_group, [csv_file], args.plot_aggregates)

        print("===== Plot assemblages =====")
        plot_several_lines_per_plot(save_dir_path_group, csv_files_in_one_group,
                                    args.plot_assemble_tags_in_one_plot, args.plot_aggregates)


def plot_one_graph(save_dir_path, csv_file_list, plot_aggregates):
    data_groups = {}
    all_tags_seen = []
    for idx, csv_file in enumerate(csv_file_list):
        df = pd.read_csv(csv_file, sep=";", index_col=0)
        tag = extract_tag_from_file_name(csv_file)
        all_tags_seen.append(tag)

        # df.index("Unnamed: 0")
        if "min_max" in plot_aggregates:
            assert "one_std" not in plot_aggregates
            df = df.rename(columns={'min': f'mean{LOWER_ENVELOPE_SUFFIX}', 'max': f'mean{UPPER_ENVELOPE_SUFFIX}'})
        else:
            df = df.drop(columns=['min', 'max'])
        if "one_std" in plot_aggregates:
            assert "min_max" not in plot_aggregates
            df[f'mean{LOWER_ENVELOPE_SUFFIX}'] = df['mean'] - df['std']
            df[f'mean{UPPER_ENVELOPE_SUFFIX}'] = df['mean'] + df['std']
        df = df.drop(columns=['std', 'var', "median"])

        data_groups[tag] = df

    plot_options = PlotConfig(xlabel="points logged in tensorboard",
                              ylabel=fing_longer_substr(all_tags_seen).strip("_"),
                              save_dir_path=save_dir_path)
    plot_helper = PlotHelper(plot_options)
    plot_helper.plot_lines(data_groups)


def plot_several_lines_per_plot(save_dir_path, csv_file_list, plot_assemble_tags_in_one_plot, plot_aggregates):

    for list_of_tags_in_assemblage in plot_assemble_tags_in_one_plot:
        # select files for one assemblage
        assemblage_list = []
        for csv_file in csv_file_list:
            if any([select_key in csv_file for select_key in list_of_tags_in_assemblage]):
                assemblage_list.append(csv_file)
        # plot one assemblage
        plot_one_graph(save_dir_path, assemblage_list, plot_aggregates)



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
    tb_data_extractor = TensorBoardDataExtractor()
    tb_data_extractor.extract_data(args.main_path, args.ignore_keys, args.group_keys, args.output)

    if args.output == "csv":
        plot_selected_keys(args)

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
