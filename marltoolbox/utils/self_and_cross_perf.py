import copy
import json
import logging
import os
import pickle
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-whitegrid')

import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer
from ray.tune.analysis import ExperimentAnalysis
from marltoolbox.utils import restore, log, miscellaneous
from marltoolbox.utils.plot import PlotHelper, PlotConfig

logger = logging.getLogger(__name__)


class SelfAndCrossPlayEvaluator:
    """
    Utility to run self-play and cross-play performance evaluation.

    Does support only the RLLib API.
    Thus if you are working with Tune, then you will need to use the utils.policy.get_tune_policy_class helper
    to convert your Tune trainer into frozen RLLib policies.
    """
    SELF_PLAY_MODE = "self-play"
    CROSS_PLAY_MODE = "cross-play"
    MODES = [CROSS_PLAY_MODE, SELF_PLAY_MODE]

    def __init__(self, exp_name: str, use_random_policy_from_own_checkpoint: bool = False):
        """
        You should take a look at examples using this class.
        Any training is deactivated here. Only the worker rollout will evaluate your policy on the environment.
        Any exploration is deactivated.

        Works for a unique pair of RLLib policies.

        :param exp_name: Normal exp_name argument provided to tune.run().
        :param use_random_policy_from_own_checkpoint: (optional, default to False)
        """
        self.default_selected_order = 0
        self.exp_name, self.exp_parent_dir = log.log_in_current_day_dir(exp_name)
        self.results_file_name = "SelfAndCrossPlay_save.p"
        self.save_path = os.path.join(self.exp_parent_dir, self.results_file_name)
        # TODO this var name is not clear enough (use_random_policy_from_own_checkpoint)
        self.use_random_policy_from_own_checkpoint = use_random_policy_from_own_checkpoint

        self.experiment_defined = False
        self.checkpoints_loaded = False

    def perform_evaluation_or_load_data(self, evaluation_config, stop_config, policies_to_load_from_checkpoint,
                                        tune_analysis_per_exp:list, TrainerClass=PGTrainer, TuneTrainerClass=None,
                                        n_cross_play_per_checkpoint:int=1, n_self_play_per_checkpoint:int=1,
                                        to_load_path:str=None):
        """

        :param evaluation_config: Normal config argument provided to tune.run().
            This RLLib config will be used to run many similar runs.
            This config will be automatically updated to load the policies from the checkpoints you are going to provide.
        :param stop_config: Normal stop_config argument provided to tune.run().
        :param policies_to_load_from_checkpoint:
        :param tune_analysis_per_exp: List of the tune_analysis you want to extract the groups of checkpoints from.
            All the checkpoints in these tune_analysis will be extracted.
        :param TrainerClass: (default is the PGTrainer class) Normal 1st argument (run_or_experiment) provided to
            tune.run(). You should use the one which provides the data flow you need. (Probably a simple PGTrainer will do).
        :param TuneTrainerClass: Will only be needed when you are going to evaluate policies created from a Tune
            trainer. You need to provide the class of this trainer.
        :param n_cross_play_per_checkpoint: (int) How many cross-play experiment per checkpoint you want to run.
            They are run randomly against the other checkpoints.
        :param n_self_play_per_checkpoint: (int) How many self-play experiment per checkpoint you want to run.
            More than 1 mean that you are going to run several times the same experiments.
        :param to_load_path: where to load the data from
        :return: data formatted in a way ready for plotting by the plot_results method.
        """
        if to_load_path is None:
            self.define_the_experiment_to_run(
                TrainerClass=TrainerClass,
                TuneTrainerClass=TuneTrainerClass,
                evaluation_config=evaluation_config,
                stop_config=stop_config,
                policies_to_load_from_checkpoint=policies_to_load_from_checkpoint,
            )
            self.preload_checkpoints_from_tune_results(tune_results=tune_analysis_per_exp)
            analysis_metrics_per_mode = self.evaluate_performances(
                n_self_play_per_checkpoint=n_self_play_per_checkpoint,
                n_cross_play_per_checkpoint=n_cross_play_per_checkpoint,
            )
        else:
            analysis_metrics_per_mode = self.load_results(to_load_path=to_load_path)
        return analysis_metrics_per_mode

    def define_the_experiment_to_run(self,
                                     evaluation_config: dict,
                                     stop_config: dict,
                                     TuneTrainerClass=None,
                                     TrainerClass=PGTrainer,
                                     policies_to_load_from_checkpoint: list = ["All"],
                                     ):
        """
        :param evaluation_config: Normal config argument provided to tune.run().
            This RLLib config will be used to run many similar runs.
            This config will be automatically updated to load the policies from the checkpoints you are going to provide.
        :param stop_config: Normal stop_config argument provided to tune.run().
        :param TuneTrainerClass: Will only be needed when you are going to evaluate policies created from a Tune
            trainer. You need to provide the class of this trainer.
        :param TrainerClass: (default is the PGTrainer class) Normal 1st argument (run_or_experiment) provided to
            tune.run(). You should use the one which provides the data flow you need. (Probably a simple PGTrainer will do).
        :param policies_to_load_from_checkpoint:
        """

        self.TrainerClass = TrainerClass
        self.TuneTrainerClass = TuneTrainerClass
        self.evaluation_config = miscellaneous.set_config_for_evaluation(evaluation_config)
        self.stop_config = stop_config

        self._config_is_for_two_players()

        self.policies_ids_sorted = sorted(list(self.evaluation_config["multiagent"]["policies"].keys()))
        self.policies_to_load_from_checkpoint = sorted(
            [policy_id for policy_id in self.policies_ids_sorted if
                                                 self._is_policy_to_load(policy_id, policies_to_load_from_checkpoint)])

        self.experiment_defined = True

    def _config_is_for_two_players(self):
        assert "multiagent" in self.evaluation_config.keys()
        assert len(self.evaluation_config["multiagent"]["policies"].keys()) == 2

    def _is_policy_to_load(self, policy_id, policies_to_load_from_checkpoint):
        return policy_id in policies_to_load_from_checkpoint or "All" in policies_to_load_from_checkpoint

    def preload_checkpoints_from_tune_results(self, tune_results: Dict[str, ExperimentAnalysis]):
        """
        :param tune_results: Dict of the tune_analysis you want to extract the groups of checkpoints from.
            All the checkpoints in these tune_analysis will be extracted.
        """
        self._extract_groups_of_checkpoints(tune_results)
        self.n_checkpoints = len(self.checkpoints)
        print(f"Found {self.n_checkpoints} checkpoints in {len(tune_results)} tune_results")

        self.checkpoints_loaded = True

    def _extract_groups_of_checkpoints(self, tune_results: Dict[str, ExperimentAnalysis]):
        self.checkpoints = []
        for idx, (group_name, one_tune_result) in enumerate(tune_results.items()):
            self._extract_one_group_of_checkpoints(idx, one_tune_result, group_name)

    def _extract_one_group_of_checkpoints(self, idx, one_tune_result: ExperimentAnalysis, group_name):
        checkpoints_in_one_group = miscellaneous.extract_checkpoints(one_tune_result)
        self.checkpoints.extend([{"group_name": group_name, "path": checkpoint}
                                 for checkpoint in checkpoints_in_one_group])

    def evaluate_performances(self, n_self_play_per_checkpoint: int, n_cross_play_per_checkpoint: int):
        """
        :param n_self_play_per_checkpoint: (int) How many self-play experiment per checkpoint you want to run.
            More than 1 mean that you are going to run several times the same experiments.
        :param n_cross_play_per_checkpoint: (int) How many cross-play experiment per checkpoint you want to run.
            They are run randomly against the other checkpoints.
        :return: data formatted in a way ready for plotting by the plot_results method.
        """
        assert self.checkpoints_loaded, "You must define the checkpoints to evaluate with the " \
                                               "preload_checkpoints_from_tune_results method"
        assert self.experiment_defined, "You must define the evaluation experiment with the " \
                                              "define_the_experiment_to_run method."

        self._validate_number_of_requested_evaluations(n_self_play_per_checkpoint, n_cross_play_per_checkpoint)
        all_metadata = self._evaluate_performances_in_parallel(n_self_play_per_checkpoint, n_cross_play_per_checkpoint)
        analysis_metrics_per_mode = self._group_results_and_extract_metrics(all_metadata)
        self.save_results(analysis_metrics_per_mode)
        return analysis_metrics_per_mode

    def _validate_number_of_requested_evaluations(self, n_self_play_per_checkpoint, n_cross_play_per_checkpoint):
        assert n_self_play_per_checkpoint + n_cross_play_per_checkpoint >= 1
        assert n_self_play_per_checkpoint >= 0
        assert n_cross_play_per_checkpoint >= 0
        if n_cross_play_per_checkpoint > 0:
            assert n_cross_play_per_checkpoint <= self.n_checkpoints - 1

    def _evaluate_performances_in_parallel(self, n_self_play_per_checkpoint, n_cross_play_per_checkpoint):
        master_config, all_metadata = self._prepare_one_master_config_dict(n_self_play_per_checkpoint,
                                                                           n_cross_play_per_checkpoint)
        results = ray.tune.run(self.TrainerClass, config=master_config,
                               stop=self.stop_config, name=os.path.join(self.exp_name, "self_and_cross_play_eval"),
                               checkpoint_freq=0, checkpoint_at_end=False)

        all_metadata_wt_results = self._add_results_into_metadata(all_metadata, results)
        return all_metadata_wt_results

    def _prepare_one_master_config_dict(self, n_self_play_per_checkpoint, n_cross_play_per_checkpoint):
        opponents_per_checkpoint = self._get_opponents_per_checkpoints(n_cross_play_per_checkpoint)
        all_config_variations, all_metadata = self._produce_config_variations(n_self_play_per_checkpoint,
                                                                              n_cross_play_per_checkpoint,
                                                                              opponents_per_checkpoint)
        master_config = self._assemble_in_one_master_config(all_config_variations)

        return master_config, all_metadata

    def _get_opponents_per_checkpoints(self, n_cross_play_per_checkpoint):
        opponents_per_checkpoint = [self._select_opponent_randomly(checkpoint_i, n_cross_play_per_checkpoint)
                                    for checkpoint_i in range(self.n_checkpoints)]
        return opponents_per_checkpoint

    def _produce_config_variations(self, n_self_play_per_checkpoint,
                                   n_cross_play_per_checkpoint, opponents_per_checkpoint):
        self_plays = [self._get_config_for_one_self_play(checkpoint_i)
                      for checkpoint_i in range(self.n_checkpoints)
                      for _ in range(n_self_play_per_checkpoint)]
        cross_plays = [self._get_config_for_one_cross_play(checkpoint_i,
                                                           opponents_per_checkpoint[checkpoint_i][cross_play_n])
                       for checkpoint_i in range(self.n_checkpoints)
                       for cross_play_n in range(n_cross_play_per_checkpoint)]
        print(f"Prepared {len(self_plays)} sself_plays and {len(cross_plays)} cross_plays")
        all_plays = self_plays + cross_plays

        all_metadata = [play[0] for play in all_plays]
        all_config_variations = [play[1] for play in all_plays]

        return all_config_variations, all_metadata

    def _assemble_in_one_master_config(self, all_config_variations):
        master_config = all_config_variations[0]
        all_multiagent_policies = [play["multiagent"]["policies"] for play in all_config_variations]
        master_config["multiagent"]["policies"] = tune.grid_search(all_multiagent_policies)
        return master_config

    def _add_results_into_metadata(self, all_metadata, results):
        for i in range(len(all_metadata)):
            all_metadata[i]["results"] = results.trials[i]
        return all_metadata

    def save_results(self, all_results):
        pickle.dump(all_results, open(self.save_path, "wb"))

    def load_results(self, to_load_path):
        assert to_load_path.endswith(self.results_file_name), f"to_load_path {to_load_path} should end with " \
                                                              f"self.results_file_name {self.results_file_name}"
        all_results = pickle.load(open(to_load_path, "rb"))
        tail, head = os.path.split(to_load_path)
        self.exp_parent_dir = tail
        return all_results

    def _get_config_for_one_self_play(self, checkpoint_i):
        metadata = {"mode": self.SELF_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Add the checkpoints to load from in the policy_config
        for policy_id in self.policies_to_load_from_checkpoint:
            metadata[policy_id] = {"checkpoint_path": self.checkpoints[checkpoint_i]["path"],
                                   "checkpoint_i": checkpoint_i}
            config_copy["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = (
                self.checkpoints[checkpoint_i]["path"], policy_id)
            config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
            config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass
        return metadata, config_copy

    def _get_config_for_one_cross_play(self, own_checkpoint_i, opponent_i):
        metadata = {"mode": self.CROSS_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        if self.use_random_policy_from_own_checkpoint:
            own_position = random.randint(0, len(config_copy["multiagent"]["policies"]) - 1)
        else:
            own_position = self.default_selected_order

        # Add the checkpoints to load from in the policy_config
        for policy_id in self.policies_to_load_from_checkpoint:
            policy_idx = self.policies_ids_sorted.index(policy_id)
            if own_position == policy_idx:
                checkpoint_idx = own_checkpoint_i
            else:
                checkpoint_idx = opponent_i
            checkpoint_path = self.checkpoints[checkpoint_idx]["path"]
            metadata[policy_id] = {"checkpoint_path": checkpoint_path,
                                   "checkpoint_i": checkpoint_idx}
            config_copy["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = (
                checkpoint_path, policy_id)
            config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
            config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass
        return metadata, config_copy

    def _select_opponent_randomly(self, checkpoint_i, n_cross_play_per_checkpoint):
        checkpoint_list_minus_i = list(range(len(self.checkpoints)))
        checkpoint_list_minus_i.pop(checkpoint_i)
        opponents = random.sample(checkpoint_list_minus_i, n_cross_play_per_checkpoint)
        return opponents

    def _split_results_per_mode_and_group_pair_id(self, all_metadata_wt_results):
        analysis_per_mode = []

        metadata_per_modes = self._split_metadata_per_mode(all_metadata_wt_results)
        for mode, metadata_for_one_mode in metadata_per_modes.items():
            analysis_per_mode.extend(self._split_metadata_per_group_pair_id(metadata_for_one_mode, mode))

        return analysis_per_mode

    def _split_metadata_per_mode(self, all_results):
        return {mode: [report for report in all_results if report["mode"] == mode] for mode in self.MODES}

    def _split_metadata_per_group_pair_id(self, metadata_for_one_mode, mode):
        analysis_per_group_pair_id = []

        tune_analysis = [metadata["results"] for metadata in metadata_for_one_mode]
        group_pair_names = [self._get_pair_of_group_names(metadata) for metadata in metadata_for_one_mode]
        group_pair_ids = [self._get_id_of_pair_of_group_names(one_pair_of_names) for one_pair_of_names in group_pair_names]
        group_pair_ids_in_this_mode = sorted(set(group_pair_ids))

        for group_pair_id in list(group_pair_ids_in_this_mode):
            filtered_analysis_list, one_pair_of_names = \
                self._find_and_group_results_for_one_group_pair_id(
                    group_pair_id, tune_analysis, group_pair_ids, group_pair_names)
            analysis_per_group_pair_id.append((mode, filtered_analysis_list, group_pair_id, one_pair_of_names))
        return analysis_per_group_pair_id

    def _find_and_group_results_for_one_group_pair_id(self,
                                                      group_pair_id, tune_analysis, group_pair_ids, group_pair_names):
        filtered_group_pair_names, filtered_tune_analysis = [], []
        for one_tune_analysis, id_, pair_of_names in zip(tune_analysis, group_pair_ids, group_pair_names):
            if id_ == group_pair_id:
                filtered_tune_analysis.append(one_tune_analysis)
                filtered_group_pair_names.append(pair_of_names)

        filtered_ids = [ self._get_id_of_pair_of_group_names(one_pair_of_names) for one_pair_of_names in filtered_group_pair_names]
        assert len(set(filtered_ids)) == 1
        one_pair_of_names = filtered_group_pair_names[0]

        return filtered_tune_analysis, one_pair_of_names

    def _extract_all_metrics(self, analysis_per_mode):
        analysis_metrics_per_mode = []
        for mode_i, mode_data in enumerate(analysis_per_mode):
            mode, analysis_list, group_pair_id, group_pair_name = mode_data

            available_metrics_list = []
            for trial in analysis_list:
                available_metrics = trial.metric_analysis
                available_metrics_list.append(available_metrics)
            analysis_metrics_per_mode.append((mode, available_metrics_list, group_pair_id, group_pair_name))
        return analysis_metrics_per_mode

    def _group_results_and_extract_metrics(self, all_metadata_wt_results):
        analysis_per_mode_per_group_pair_id = self._split_results_per_mode_and_group_pair_id(all_metadata_wt_results)
        analysis_metrics_per_mode_per_group_pair_id = self._extract_all_metrics(analysis_per_mode_per_group_pair_id)
        return analysis_metrics_per_mode_per_group_pair_id

    def _get_id_of_pair_of_group_names(self, pair_of_group_names):
        id_of_pair_of_group_names = ''.join(pair_of_group_names)
        return id_of_pair_of_group_names

    def _get_pair_of_group_names(self, metadata):
        checkpoints_idx_used = [metadata[policy_id]["checkpoint_i"]
                                for policy_id in self.policies_to_load_from_checkpoint]
        pair_of_group_names = [self.checkpoints[checkpoint_i]["group_name"] for checkpoint_i in checkpoints_idx_used]
        return pair_of_group_names

    def plot_results(self, analysis_metrics_per_mode, plot_config, x_axis_metric, y_axis_metric):
        plotter = SelfAndCrossPlayPlotter()
        plotter.plot_results(exp_parent_dir=self.exp_parent_dir,
                             metrics_per_mode=analysis_metrics_per_mode, plot_config=plot_config,
                             x_axis_metric=x_axis_metric, y_axis_metric=y_axis_metric)


class SelfAndCrossPlayPlotter:
    def __init__(self):
        self.x_axis_metric, self.y_axis_metric, self.metric_mode = None, None, None
        self.stat_summary = None
        self.data_groups_per_mode = None

    def plot_results(self,
                     exp_parent_dir: str,
                     x_axis_metric: str,
                     y_axis_metric: str,
                     metrics_per_mode: list,
                     plot_config: PlotConfig,
                     metric_mode: str = "avg",
                     ):
        self._reset(x_axis_metric, y_axis_metric, metric_mode)
        for metrics_for_one_evaluation_mode in metrics_per_mode:
            self._extract_performance_evaluation_points(metrics_for_one_evaluation_mode)
        self.stat_summary.save_summary(filename_prefix="self_and_cross_play", folder_dir=exp_parent_dir)
        self. _plot_and_save_fig(plot_config, exp_parent_dir)

    def _reset(self, x_axis_metric, y_axis_metric, metric_mode):
        self.x_axis_metric, self.y_axis_metric, self.metric_mode = x_axis_metric, y_axis_metric, metric_mode
        self.stat_summary = StatisticSummary(self.x_axis_metric, self.y_axis_metric, self.metric_mode)
        self.data_groups_per_mode = {}

    def _extract_performance_evaluation_points(self, metrics_for_one_evaluation_mode):
        (mode, available_metrics_list, group_pair_id, group_pair_name) = metrics_for_one_evaluation_mode

        label = self._get_label(mode, group_pair_name)
        x, y = self._extract_x_y_points(available_metrics_list)

        self.stat_summary.aggregate_stats_on_data_points(x, y, label)
        self.data_groups_per_mode[label] = self._format_as_df(x, y)
        print("x, y", x, y)

    def _get_label(self, mode, group_pair_name):
        # For backward compatibility
        if mode == "Same-play" or mode == "same training run":
            mode = self.SELF_PLAY_MODE
        elif mode == "Cross-play" or mode == "cross training run":
            mode = self.CROSS_PLAY_MODE

        print("Evaluator mode:", mode)
        if self._suffix_needed(group_pair_name):
            print("Using group_pair_name:", group_pair_name)
            label = f"{mode}: " + " vs ".join(group_pair_name)
        else:
            label = mode
        label = label.replace('_', ' ')
        print("label",label)
        return label

    def _suffix_needed(self, group_pair_name):
        return group_pair_name is not None and \
                all([name is not None for name in group_pair_name]) and \
                all(group_pair_name)

    def _extract_x_y_points(self, available_metrics_list):
        x, y = [], []
        assert len(available_metrics_list) > 0
        random.shuffle(available_metrics_list)
        for available_metrics in available_metrics_list:
            x_point = available_metrics[self.x_axis_metric][self.metric_mode]
            y_point = available_metrics[self.y_axis_metric][self.metric_mode]
            x.append(x_point)
            y.append(y_point)
        return x, y

    def _format_as_df(self, x, y):
        group_df_dict = {"": [(one_x_point, one_y_point) for one_x_point, one_y_point in zip(x, y)]}
        group_df = pd.DataFrame(group_df_dict)
        return group_df

    def _plot_and_save_fig(self, plot_config, exp_parent_dir):
        plot_helper = PlotHelper(plot_config)
        plot_helper.plot_cfg.save_dir_path = exp_parent_dir
        plot_helper.plot_dots(self.data_groups_per_mode)

class StatisticSummary:

    def __init__(self, x_axis_metric, y_axis_metric, metric_mode):
        self.x_means, self.x_se, self.x_labels, self.x_raw = [], [], [], []
        self.y_means, self.y_se, self.y_labels, self.y_raw = [], [], [], []
        self.matrix_label = []
        self.x_axis_metric, self.y_axis_metric = x_axis_metric, y_axis_metric
        self.metric_mode = metric_mode

    def aggregate_stats_on_data_points(self, x, y, label):
        self.x_means.append(sum(x) / len(x))
        self.x_se.append(np.array(x).std() / np.sqrt(len(x)))
        self.x_labels.append(f"Metric:{self.x_axis_metric}, Metric mode:{self.metric_mode}")

        self.y_means.append(sum(y) / len(y))
        self.y_se.append(np.array(y).std() / np.sqrt(len(y)))
        self.y_labels.append(f"Metric:{self.y_axis_metric}, Metric mode:{self.metric_mode}")

        self.matrix_label.append(label)
        self.x_raw.append(x)
        self.y_raw.append(y)

    def save_summary(self, filename_prefix, folder_dir):
        file_name = f'{filename_prefix}_{self.y_axis_metric}_vs_{self.x_axis_metric}_matrix.json'
        file_name = file_name.replace('/', '_')
        file_path = os.path.join(folder_dir, file_name)
        formated_data = {}
        for x_mean, x_std_err, x_lbl, y_mean, y_std_err, y_lbl, lbl, x, y in zip(
                self.x_means, self.x_se, self.x_labels, self.y_means, self.y_se,
                self.y_labels, self.matrix_label, self.x_raw, self.y_raw):
            formated_data[lbl] = {
                x_lbl: {"mean": x_mean, "std_err": x_std_err, "raw_data": str(x)},
                y_lbl: {"mean": y_mean, "std_err": y_std_err, "raw_data": str(y)}
            }
        with open(file_path, 'w') as f:
            json.dump(formated_data, f, indent=4, sort_keys=True)


