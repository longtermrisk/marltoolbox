# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# Improved version of the original GitHub project:
# https://github.com/Spenhouet/tensorboard-aggregator
# ===========================================================================

import argparse
import ast
import datetime
import os
import re
from typing import Iterable

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from marltoolbox.utils.miscellaneous import (
    ignore_str_containing_keys,
    separate_str_in_group_containing_keys,
    GROUP_KEY_NONE,
    keep_strs_containing_keys,
    fing_longer_substr,
)
from marltoolbox.utils.path import list_all_files_in_one_dir_tree
from marltoolbox.utils.plot import (
    LOWER_ENVELOPE_SUFFIX,
    UPPER_ENVELOPE_SUFFIX,
    PlotHelper,
    PlotConfig,
)

FOLDER_NAME = "aggregates"
AGGREGATION_OPS = {
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "median": np.median,
    "std": np.std,
    "var": np.var,
}
COLORS = list(mcolors.TABLEAU_COLORS)

PLOT_KEYS = [
    "grad_gnorm",
    "reward",
    "loss",
    "entropy",
    "entropy_avg",
    "td_error",
    "error",
    "act_dist_inputs_avg",
    "act_dist_inputs_single",
    "q_values_avg",
    "action_prob",
    "q_values_single",
    "_lr",
    "max_q_values",
    "min_q_values",
    "learn_on_batch",
    "timers",
    "ms",
    "throughput",
    "temperature",
]

PLOT_ASSEMBLAGE_TAGS = [
    ("policy_reward_mean",),
    ("grad_gnorm",),
    ("entropy_buffer_samples_avg",),
    ("entropy_avg",),
    ("loss",),
    ("td_error",),
    ("loss", "td_error"),
    ("learn_on_batch",),
    ("last_training_max_q_values",),
    ("last_training_min_q_values",),
    ("act_dist_inputs_avg_act0",),
    ("act_dist_inputs_avg_act1",),
    ("act_dist_inputs_avg_act2",),
    ("act_dist_inputs_avg_act3",),
    ("q_values_avg_act0",),
    ("q_values_avg_act1",),
    ("q_values_avg_act2",),
    ("q_values_avg_act3",),
    ("q_values_single_max",),
    ("act_dist_inputs_single_max",),
    ("action_prob_single",),
    ("action_prob_avg",),
    ("reward",),
    ("last_training_max_q_values", "last_training_target_max_q_values"),
    ("last_training_min_q_values", "last_training_target_min_q_values"),
    ("timers",),
    ("ms",),
    ("throughput",),
    ("_lr",),
    ("temperature",),
]


class TensorBoardDataExtractor:
    def __init__(self, main_path):
        self.main_path = main_path

        now = datetime.datetime.now()
        self.date_hour_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        save_dir = os.path.join(self.main_path, FOLDER_NAME)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, self.date_hour_str)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir

    def extract_data(self, ignore_keys, group_keys, output):
        print("\n===== Extract data =====")
        file_list = list_all_files_in_one_dir_tree(self.main_path)
        file_list_filtered = ignore_str_containing_keys(file_list, ignore_keys)
        file_list_dict = separate_str_in_group_containing_keys(
            file_list_filtered, group_keys
        )

        self._aggregate(self.main_path, output, file_list_dict)
        return self.save_dir

    def _aggregate(self, main_path, output, file_list_dict):

        print(f"Started aggregation {main_path}")

        extracts_per_group = {
            group_key: self._extract_x_y_per_keys(main_path, file_list)
            for group_key, file_list in file_list_dict.items()
        }

        if output == "summary":
            raise NotImplementedError()
            # To implement it, you can look at the original implementation at
            # https://github.com/Spenhouet/tensorboard-aggregator
        elif output == "csv":
            self._aggregate_to_csv(main_path, AGGREGATION_OPS, extracts_per_group)

        print(f"End of aggregation {main_path}")

    def _extract_x_y_per_keys(self, main_path, file_list):
        print(
            "Going to extract",
            main_path,
            "with len(file_list)",
            len(file_list),
        )

        event_readers = self._create_event_reader_for_each_log_files(file_list)
        if len(event_readers) == 0:
            return None
        (
            all_scalar_events_per_key,
            keys,
        ) = self._get_and_validate_all_scalar_keys(event_readers)
        (
            steps_per_key,
            all_scalar_events_per_key,
        ) = self._get_and_validate_all_steps_per_key(all_scalar_events_per_key, keys)
        values_per_key = self._get_values_per_step_per_key(all_scalar_events_per_key)

        keys = [key.replace("/", "_") for key in keys]
        all_per_key = dict(zip(keys, zip(steps_per_key, values_per_key)))

        return all_per_key

    def _create_event_reader_for_each_log_files(self, file_list):
        event_readers = [
            EventAccumulator(file_path).Reload().scalars for file_path in file_list
        ]

        # Filter non event files
        event_readers = [
            one_event_reader
            for one_event_reader in event_readers
            if one_event_reader.Keys()
        ]
        print(f"found {len(event_readers)} event_readers")
        return event_readers

    def _get_and_validate_all_scalar_keys(self, event_readers):

        all_keys = [
            tuple(one_event_reader.Keys()) for one_event_reader in event_readers
        ]

        self._print_discrepencies_in_keys(all_keys)
        keys = self._get_common_keys(all_keys)

        all_scalar_events_per_key = [
            [one_event_reader.Items(key) for one_event_reader in event_readers]
            for key in keys
        ]
        return all_scalar_events_per_key, keys

    def _print_discrepencies_in_keys(self, all_keys):
        if len(set(all_keys)) != 1:
            missing_k_detected = []
            for keys_1 in all_keys:
                for keys_2 in all_keys:
                    for k in keys_1:
                        if k not in keys_2:
                            if k not in missing_k_detected:
                                print(
                                    f"key {k} is not present in all " f"event_readers"
                                )
                                missing_k_detected.append(k)

    def _get_common_keys(self, all_keys):
        common_keys = []
        for one_key in all_keys[0]:
            common = True
            for keys in all_keys:
                if one_key not in keys:
                    if one_key not in keys:
                        common = False
                        break
            if common:
                common_keys.append(one_key)
        return common_keys

    def _get_and_validate_all_steps_per_key(self, all_scalar_events_per_key, keys):
        all_steps_per_key = [
            [
                tuple(scalar_event.step for scalar_event in scalar_events)
                for scalar_events in all_scalar_events
            ]
            for all_scalar_events in all_scalar_events_per_key
        ]

        steps_per_key = []
        for key_idx, (all_steps_for_one_key, key) in enumerate(
            zip(all_steps_per_key, keys)
        ):
            self._print_discrepencies_in_steps(all_steps_for_one_key, key)
            common_steps = self._keep_common_steps(all_steps_for_one_key)
            all_scalar_events_per_key = self._remove_events_if_step_missing_somewhere(
                common_steps, all_scalar_events_per_key, key_idx
            )
            steps_per_key.append(common_steps)

        return steps_per_key, all_scalar_events_per_key

    def _print_discrepencies_in_steps(self, all_steps_for_one_key, key):
        for steps_1 in all_steps_for_one_key:
            for steps_2 in all_steps_for_one_key:
                missing_steps = [step for step in steps_1 if step not in steps_2]
                if len(missing_steps) > 0:
                    print(
                        f"discrepency in steps logged for key {key}:"
                        f"{missing_steps} missing"
                    )
                    break

    def _keep_common_steps(self, all_steps_for_one_key):
        common_steps = []
        for one_step in all_steps_for_one_key[0]:
            common = True
            for keys in all_steps_for_one_key:
                if one_step not in keys:
                    if one_step not in keys:
                        common = False
                        break
            if common:
                common_steps.append(one_step)
        return common_steps

    def _remove_events_if_step_missing_somewhere(
        self, common_steps, all_scalar_events_per_key, key_idx
    ):

        all_scalar_events_per_key[key_idx] = [
            [
                scalar_event
                for scalar_event in scalar_events_batch
                if scalar_event.step in common_steps
            ]
            for scalar_events_batch in all_scalar_events_per_key[key_idx]
        ]
        return all_scalar_events_per_key

    def _get_values_per_step_per_key(self, all_scalar_events_per_key):
        values_per_key = [
            [
                [scalar_event.value for scalar_event in scalar_events]
                for scalar_events in all_scalar_events
            ]
            for all_scalar_events in all_scalar_events_per_key
        ]
        return values_per_key

    def _aggregate_to_csv(self, main_path, aggregation_ops, extracts_per_group):
        for group_key, all_per_key in extracts_per_group.items():
            if all_per_key is None:
                continue
            for key, (steps, values) in all_per_key.items():
                aggregations = {
                    key: aggregation_operation(values, axis=0)
                    for key, aggregation_operation in aggregation_ops.items()
                }
                if "reward" in key and "mean" in key:
                    aggregations["values"] = [
                        str(el) for el in np.array(values).T.tolist()
                    ]
                self._write_csv(main_path, group_key, key, aggregations, steps)

    def _write_csv(self, main_path, group_key, key, aggregations, steps):
        main_path_split = os.path.split(main_path)
        group_dir = self._get_valid_filename(group_key)
        save_group_dir = (
            os.path.join(self.save_dir, group_dir)
            if group_key != GROUP_KEY_NONE
            else self.save_dir
        )
        if not os.path.exists(save_group_dir):
            os.mkdir(save_group_dir)
        file_name = self._get_valid_filename(key) + "-" + main_path_split[-1] + ".csv"
        df = pd.DataFrame(aggregations, index=steps)
        save_dir_file_path = os.path.join(save_group_dir, file_name)
        df.to_csv(save_dir_file_path, sep=";")

    def _get_valid_filename(self, s):
        s = str(s).strip().replace(" ", "_")
        return re.sub(r"(?u)[^-\w.]", "", s)


class SummaryPlotter:
    def plot_selected_keys(
        self,
        save_dir,
        plot_keys,
        group_keys,
        plot_aggregates,
        plot_assemble_tags_in_one_plot,
        plot_single_lines,
        plot_labels_cleaning,
        additional_plot_config_kwargs,
    ):

        self.plot_labels_cleaning = plot_labels_cleaning
        self.plot_aggregates = plot_aggregates
        self.additional_plot_config_kwargs = additional_plot_config_kwargs

        print("\n===== Plot =====")
        save_dir_path = save_dir
        file_list = list_all_files_in_one_dir_tree(save_dir_path)
        file_list = keep_strs_containing_keys(file_list, plot_keys)
        csv_file_list = [file_path for file_path in file_list if "csv" in file_path]
        csv_file_groups = separate_str_in_group_containing_keys(
            csv_file_list, group_keys
        )

        for group_key, csv_files_in_one_group in csv_file_groups.items():
            save_dir_path_group = (
                os.path.join(save_dir_path, group_key)
                if group_key != GROUP_KEY_NONE
                else save_dir_path
            )

            if not os.path.exists(save_dir_path_group):
                os.mkdir(save_dir_path_group)

            print("===== Plot simple plots =====")
            if plot_single_lines:
                for csv_file in csv_files_in_one_group:
                    self.plot_one_graph(save_dir_path_group, [csv_file])

            print("===== Plot assemblages =====")
            self.plot_several_lines_per_plot(
                save_dir_path_group,
                csv_files_in_one_group,
                plot_assemble_tags_in_one_plot,
            )

    def plot_one_graph(self, save_dir_path, csv_file_list, y_label=None):
        data_groups = {}
        all_tags_seen = []
        for _, csv_file in enumerate(csv_file_list):
            df = pd.read_csv(csv_file, sep=";", index_col=0)
            tag = self.extract_tag_from_file_name(csv_file)
            all_tags_seen.append(tag)

            if "min_max" in self.plot_aggregates:
                assert "one_std" not in self.plot_aggregates
                df = df.rename(
                    columns={
                        "min": f"mean{LOWER_ENVELOPE_SUFFIX}",
                        "max": f"mean{UPPER_ENVELOPE_SUFFIX}",
                    }
                )
            else:
                df = df.drop(columns=["min", "max"])
            if "one_std" in self.plot_aggregates:
                assert "min_max" not in self.plot_aggregates
                df[f"mean{LOWER_ENVELOPE_SUFFIX}"] = df["mean"] - df["std"]
                df[f"mean{UPPER_ENVELOPE_SUFFIX}"] = df["mean"] + df["std"]
            df = df.drop(columns=["std", "var", "median"])
            if "values" in df.columns:
                df = df.drop(columns=["values"])

            data_groups[tag] = df

        plot_options = PlotConfig(
            xlabel="steps",
            ylabel=fing_longer_substr(all_tags_seen).strip("_")
            if y_label is None
            else y_label,
            save_dir_path=save_dir_path,
            **self.additional_plot_config_kwargs,
        )
        plot_helper = PlotHelper(plot_options)
        plot_helper.plot_lines(data_groups)

    def plot_one_x_y_plot(self, save_dir_path, csv_file_list, y_label=None):
        data_groups = {}
        all_tags_seen = []

        df = pd.read_csv(csv_file_list[0], sep=";", index_col=0)
        tag = self.extract_tag_from_file_name(csv_file_list[0])
        all_tags_seen.append(tag)
        if "values" not in df.columns:
            return None
        last_values = df["values"].iloc[-1]
        X = [
            float(el.replace("[", "").replace("]", "").strip())
            for el in last_values.split(",")
        ]

        df = pd.read_csv(csv_file_list[1], sep=";", index_col=0)
        # tag = self.extract_tag_from_file_name(csv_file_list[1])
        all_tags_seen.append(tag)
        if "values" not in df.columns:
            return None
        last_values = df["values"].iloc[-1]
        Y = [
            float(el.replace("[", "").replace("]", "").strip())
            for el in last_values.split(",")
        ]

        values = [[(x, y)] for x, y in zip(X, Y)]
        # Only keep the last point of the training
        df = pd.DataFrame(data=values, columns=["mean_reward_pl"])

        data_groups[tag] = df

        plot_options = PlotConfig(
            xlabel="steps",
            ylabel=fing_longer_substr(all_tags_seen).strip("_")
            if y_label is None
            else y_label,
            save_dir_path=save_dir_path,
            filename_prefix="dot_plot",
            **self.additional_plot_config_kwargs,
        )
        plot_helper = PlotHelper(plot_options)
        plot_helper.plot_dots(data_groups)

    def plot_several_lines_per_plot(
        self, save_dir_path, csv_file_list, plot_assemble_tags_in_one_plot
    ):

        for assemblage_idx, list_of_tags_in_assemblage in enumerate(
            plot_assemble_tags_in_one_plot
        ):
            assert isinstance(list_of_tags_in_assemblage, Iterable)
            # select files for one assemblage
            assemblage_list = self._group_csv_file_in_aggregates(
                csv_file_list, list_of_tags_in_assemblage
            )

            if len(assemblage_list) > 0:
                # plot one assemblage
                y_label = f"{assemblage_idx}_" + " or ".join(list_of_tags_in_assemblage)
                self.plot_one_graph(save_dir_path, assemblage_list, y_label=y_label)
                self.plot_one_x_y_plot(save_dir_path, assemblage_list, y_label=y_label)

    def _group_csv_file_in_aggregates(self, csv_file_list, list_of_tags_in_assemblage):
        print(f"Start the {list_of_tags_in_assemblage} assemblage")
        assemblage_list = []
        for csv_file in csv_file_list:
            if any(
                [
                    select_key in os.path.split(csv_file)[-1]
                    for select_key in list_of_tags_in_assemblage
                ]
            ):
                assemblage_list.append(csv_file)
        # print("csv files selected for assemblage", assemblage_list)
        assemblage_list = sorted(assemblage_list)
        return assemblage_list

    def extract_tag_from_file_name(self, csv_file):
        filename = os.path.split(csv_file)[-1]
        path_split = filename.split("-")[0]
        path_split = path_split.split("_")
        if path_split[0] == "ray":
            path_split = path_split[1:]
        if path_split[0] == "tune":
            path_split = path_split[1:]
        tag = "_".join(path_split)

        for to_remove, replacement in self.plot_labels_cleaning:
            tag = tag.replace(to_remove, replacement)

        return tag


def add_summary_plots(
    main_path: str,
    ignore_keys: Iterable = ("aggregates", "same_cross_play"),
    group_keys: Iterable = (),
    output: str = "csv",
    plot_keys: Iterable = (
        "policy_reward_mean",
        "loss",
        "entropy",
        "entropy_avg",
        "td_error",
    ),
    plot_aggregates: Iterable = ("mean", "min_max"),
    plot_assemble_tags_in_one_plot=(
        ("policy_reward_mean",),
        ("loss", "td_error"),
        ("entropy",),
        ("entropy_avg",),
    ),
    plot_single_lines=False,
    plot_labels_cleaning: Iterable = (
        ("learner_stats_", ""),
        ("info_learner_", ""),
        ("player_", "pl_"),
    ),
    additional_plot_config_kwargs={
        "figsize": (8, 8),
        "legend_fontsize": "small",
    },
):
    """
    Aggregates multiple tensorboard runs into mean, min, max, median, std and
    save that in tensorboard files or in csv.
    Also allows to plot the csv files created.

    Usage example in a terminal:

    The TensorBoard log files are in subfolders inside
    /LOLA_PG/2021_02_02/13_02_56 then:
    python aggregate_and_plot_tensorboard_data.py
        --main_path /home/user/vm-data/ALGO_0/2021_02_02/13_02_56
        --plot_keys  [\'reward\'] --plot_aggregates  [\'mean\',\'min_max\']
        --plot_assemble_tags_in_one_plot [[\'reward_\']]

    To aggregate and plot different experiments at
    the same time (but independently) you can use the group_keys.
    group_keys (needs to be unique and in the path of each group):
    python aggregate_and_plot_tensorboard_data.py
        --main_path /home/user/vm-data/
        --group_keys [\'09_18_50\',\'13_02_56\']
        --plot_keys  [\'reward\']
        --plot_aggregates  [\'mean\',\'min_max\']
        --plot_assemble_tags_in_one_plot [[\'reward_\']]

    :param plot_single_lines:
    :param main_path: Root of the tree where to look for the
    tensorboard files.
    :param ignore_keys: Ignore the files whose path contains any of these
    strings. Useful to ignore part of the files in the tree of the main_path.
    :param group_keys: Separe the files into groups of files. Useful when
    processing several unrelated experiments at the same time.
    Keys used to separate files in groups.
    :param output: aggregation can be saves as tensorboard file (summary)
    or as table (csv)
    :param plot_keys: Only keep csv files whose path contain any of these str.
    Keys used to select tensorboard tags to plot.
    :param plot_aggregates: which results of aggregation operations to plot
    :param plot_assemble_tags_in_one_plot:
    :param plot_single_lines: Boolean to plot or not the single lines plots.
    :param plot_labels_cleaning: Iterable containing pairs of strings. For
    each pairs, the first str will be replaced by the str in all plots labels.
    :param additional_plot_config_kwargs: additional config kwargs for
    PlotConfig.
    :return:
    """

    if output not in ["summary", "csv"]:
        raise ValueError(
            "output must be one of ['summary', 'csv']" f"current output: {output}"
        )

    main_path = os.path.expanduser(main_path)

    tb_data_extractor = TensorBoardDataExtractor(main_path)
    save_dir = tb_data_extractor.extract_data(ignore_keys, group_keys, output)

    if output == "csv":
        plotter = SummaryPlotter()
        plotter.plot_selected_keys(
            save_dir,
            plot_keys,
            group_keys,
            plot_aggregates,
            plot_assemble_tags_in_one_plot,
            plot_single_lines,
            plot_labels_cleaning,
            additional_plot_config_kwargs,
        )


def param_list(param):
    p_list = ast.literal_eval(param)
    if type(p_list) is not list:
        raise argparse.ArgumentTypeError(f"Parameter {param} is not a list")
    return p_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Args for aggregation data
    parser.add_argument(
        "--main_path",
        type=str,
        help="main path for tensorboard files",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--group_keys",
        type=param_list,
        help="keys used to separate files in groups",
        default=[],
    )
    parser.add_argument(
        "--ignore_keys",
        type=param_list,
        help="keys used to ignore files",
        default=["aggregates", "same_cross_play"],
    )
    parser.add_argument(
        "--output",
        type=str,
        help="aggregation can be saves as "
        "tensorboard file (summary) or as table (csv)",
        default="csv",
    )
    # Args for plotting
    parser.add_argument(
        "--plot_keys",
        type=param_list,
        help="keys used to select tensorboard tags to plot",
        default=["reward", "loss", "entropy"],
    )
    parser.add_argument(
        "--plot_aggregates",
        type=param_list,
        help="which results of aggregation operations to plot",
        default=["mean", "min_max"],
    )
    parser.add_argument(
        "--plot_assemble_tags_in_one_plot",
        type=param_list,
        help="keys used to select tensorboard tags to " "aggregated plots",
        default=[["reward"]],
    )

    args = parser.parse_args()
    add_summary_plots(
        args.main_path,
        args.ignore_keys,
        args.group_keys,
        args.output,
        args.plot_keys,
        args.plot_aggregates,
        args.plot_assemble_tags_in_one_plot,
    )
