import copy
from collections import Iterable

import json
import logging
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from typing import Tuple

plt.style.use('seaborn-whitegrid')

import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer

from marltoolbox.utils import restore, log, miscellaneous

logger = logging.getLogger(__name__)


class SameAndCrossPlayEvaluation:
    """
    Does support using RLLib policies only or
    RLLib Policies which use internally a TuneTrainerCLass
    """
    # TODO add function to list all available metrics
    # TODO docstring
    SELF_PLAY_MODE = "same-play"
    CROSS_PLAY_MODE = "cross-play"
    MODES = [CROSS_PLAY_MODE, SELF_PLAY_MODE]

    def __init__(self, evaluation_config: dict, stop_config: dict, exp_name: str,
                 TuneTrainerClass=None,
                 TrainerClass=None,
                 checkpoint_list_group_idx: list = None,
                 group_names: list = None,
                 mix_policy_order: bool = False,
                 policies_to_train: list = ["None"],
                 policies_to_load_from_checkpoint: list = ["All"],
                 ):
        """
        Work for a unique pair of RLLib policies
        """

        self.mix_policy_order = mix_policy_order
        self.policies_to_train = policies_to_train
        self.policies_to_load_from_checkpoint = policies_to_load_from_checkpoint
        self.group_names = group_names
        self.checkpoint_list_group_idx = checkpoint_list_group_idx
        self.default_selected_order = 0

        if TrainerClass is None:
            TrainerClass = PGTrainer
        self.TrainerClass = TrainerClass
        self.TuneTrainerClass = TuneTrainerClass

        assert "multiagent" in evaluation_config.keys()
        assert len(evaluation_config["multiagent"]["policies"].keys()) == 2
        evaluation_config = miscellaneous.set_config_for_evaluation(evaluation_config, policies_to_train)
        self.evaluation_config = evaluation_config
        self.stop_config = stop_config

        self.exp_name, self.exp_parent_dir = log.log_in_current_day_dir(exp_name)
        self.results_file_name = "SameAndCrossPlay_save.p"
        self.save_path = os.path.join(self.exp_parent_dir, self.results_file_name)

        self.policies_ids_sorted = sorted(list(self.evaluation_config["multiagent"]["policies"].keys()))
        self.policies_to_load_from_checkpoint = [policy_id
                                                 for policy_id in self.policies_ids_sorted
                                                 if self._check_policy_to_load(policy_id)]

    def _init_info_about_checkpoints(self, checkpoint_list, extract_checkpoints_from_results,
                                     checkpoint_list_group_idx, group_names):
        assert extract_checkpoints_from_results or checkpoint_list
        if extract_checkpoints_from_results is not None:
            assert checkpoint_list is None and checkpoint_list_group_idx is None
            assert isinstance(extract_checkpoints_from_results, list)
            checkpoint_list = []
            checkpoint_list_group_idx = []
            for idx, result in enumerate(extract_checkpoints_from_results):
                checkpoints = miscellaneous.extract_checkpoints(result)
                group_idx = [idx] * len(checkpoints)
                checkpoint_list.extend(checkpoints)
                checkpoint_list_group_idx.extend(group_idx)
        else:
            # By default put every checkpoint in the same group
            if checkpoint_list_group_idx is None:
                checkpoint_list_group_idx = [0] * len(checkpoint_list)

        if len(set(checkpoint_list_group_idx)) > 1:
            assert group_names is not None

        self.checkpoint_list = checkpoint_list
        self.checkpoint_list_group_idx = checkpoint_list_group_idx
        self.n_checkpoints = len(self.checkpoint_list)
        self.group_names = group_names

    def _check_policy_to_load(self, policy_id):
        return policy_id in self.policies_to_load_from_checkpoint or "All" in self.policies_to_load_from_checkpoint

    def perf_analysis(self, n_same_play_per_checkpoint: int, n_cross_play_per_checkpoint: int,
                      checkpoint_list: list = None, extract_checkpoints_from_results: list = None):

        self._init_info_about_checkpoints(checkpoint_list, extract_checkpoints_from_results,
                                          self.checkpoint_list_group_idx, self.group_names)

        assert n_same_play_per_checkpoint + n_cross_play_per_checkpoint >= 1
        assert n_same_play_per_checkpoint >= 0
        assert n_cross_play_per_checkpoint >= 0
        if n_cross_play_per_checkpoint > 0:
            assert n_cross_play_per_checkpoint <= self.n_checkpoints - 1

        master_config, all_metadata = self._prepare_master_config(n_cross_play_per_checkpoint,
                                                                  n_same_play_per_checkpoint)
        results = ray.tune.run(self.TrainerClass, config=master_config,
                               stop=self.stop_config, name=os.path.join(self.exp_name, "same_cross_play"),
                               checkpoint_freq=0, checkpoint_at_end=False)
        for i in range(len(all_metadata)):
            all_metadata[i]["results"] = results.trials[i]

        analysis_metrics_per_mode = self._prepare_to_plot(all_metadata)
        self.save_results(analysis_metrics_per_mode)
        return analysis_metrics_per_mode

    def _prepare_master_config(self, n_cross_play_per_checkpoint, n_same_play_per_checkpoint):
        opponents_list = [self._select_opp_randomly(checkpoint_i, n_cross_play_per_checkpoint)
                          for checkpoint_i in range(self.n_checkpoints)]
        same_play = [self._get_config_for_one_same_play(checkpoint_i)
                     for checkpoint_i in range(self.n_checkpoints)
                     for same_play_n in range(n_same_play_per_checkpoint)]
        cross_play = [self._get_config_for_one_cross_play(checkpoint_i, opponents_list[checkpoint_i][cross_play_n])
                      for checkpoint_i in range(self.n_checkpoints)
                      for cross_play_n in range(n_cross_play_per_checkpoint)]
        all_play = same_play + cross_play
        all_metadata = [play[0] for play in all_play]
        all_config = [play[1] for play in all_play]
        master_config = all_config[0]
        all_multiagent_policies = [play["multiagent"]["policies"] for play in all_config]
        master_config["multiagent"]["policies"] = tune.grid_search(all_multiagent_policies)

        return master_config, all_metadata

    def save_results(self, all_results):
        pickle.dump(all_results, open(self.save_path, "wb"))

    def load_results(self, to_load_path):
        assert to_load_path.endswith(self.results_file_name), f"to_load_path {to_load_path} should end with " \
                                                              f"self.results_file_name {self.results_file_name}"
        all_results = pickle.load(open(to_load_path, "rb"))
        tail, head = os.path.split(to_load_path)
        self.exp_parent_dir = tail
        return all_results

    def _get_config_for_one_same_play(self, checkpoint_i):
        metadata = {"mode": self.SELF_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Add the checkpoints to load from in the policy_config
        for policy_id in self.policies_to_load_from_checkpoint:
            metadata[policy_id] = {"checkpoint_path": self.checkpoint_list[checkpoint_i],
                                   "checkpoint_i": checkpoint_i}
            config_copy["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = (
                self.checkpoint_list[checkpoint_i], policy_id)
            config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
            config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass
        return metadata, config_copy

    def _get_config_for_one_cross_play(self, own_checkpoint_i, opponent_i):
        metadata = {"mode": self.CROSS_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        if self.mix_policy_order:
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
            checkpoint_path = self.checkpoint_list[checkpoint_idx]
            metadata[policy_id] = {"checkpoint_path": checkpoint_path,
                                   "checkpoint_i": checkpoint_idx}
            config_copy["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = (
                checkpoint_path, policy_id)
            config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
            config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass
        return metadata, config_copy

    def _select_opp_randomly(self, checkpoint_i, n_cross_play_per_checkpoint):
        checkpoint_list_minus_i = list(range(len(self.checkpoint_list)))
        checkpoint_list_minus_i.pop(checkpoint_i)
        opponents = random.sample(checkpoint_list_minus_i, n_cross_play_per_checkpoint)
        return opponents

    def _split_results_per_mode_and_pair_id(self, all_results):
        analysis_per_mode = []
        # Split per modes
        for mode in self.MODES:
            reports_in_mode = [report for report in all_results if report["mode"] == mode]
            analysis_list = [report["results"] for report in reports_in_mode]
            analysis_groups_ids = [self._get_group_pair_id(report) for report in reports_in_mode]
            analysis_groups_names = [self._get_group_pair_names(report) for report in reports_in_mode]
            group_pair_ids = set(analysis_groups_ids)
            # Split per group_pair id (filter per id inside mode)
            for group_pair_id in list(group_pair_ids):
                filtered_names, filtered_ids, filtered_analysis_list = [], [], []
                for analysis, id, names in zip(analysis_list, analysis_groups_ids, analysis_groups_names):
                    if id == group_pair_id:
                        filtered_analysis_list.append(analysis)
                        filtered_ids.append(id)
                        filtered_names.append(names)
                analysis_per_mode.append((mode, filtered_analysis_list, filtered_ids[0], filtered_names[0]))
        return analysis_per_mode

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

    def _prepare_to_plot(self, all_results):
        analysis_per_mode = self._split_results_per_mode_and_pair_id(all_results)
        analysis_metrics_per_mode = self._extract_all_metrics(analysis_per_mode)
        return analysis_metrics_per_mode

    # TODO may be make the plotting independant to be used for other stuff as well
    def plot_results(self, analysis_metrics_per_mode: dict, metrics: Tuple[Tuple[str, str]], metric_mode: str = "avg",
                     alpha=1.0, colors=None, x_limits: Iterable = None, y_limits: Iterable = None,
                     scale_multipliers: Iterable = None, show=False, save_fig=True,
                     figsize=(6, 6), markersize=5, jitter=None, title_sufix="",
                     save_matrix=True, xlabel=None, ylabel=None, add_title=True, frameon=None,
                     show_groups=True, plot_max_n_points=None):

        colors = list(matplotlib.colors.BASE_COLORS.keys()) if colors is None else colors
        if self.checkpoint_list_group_idx is not None and len(set(self.checkpoint_list_group_idx)) > 1:
            n_possible_group_pairs = sum(range(len(set(self.checkpoint_list_group_idx)) + 1))
            assert len(colors) >= n_possible_group_pairs

        assert isinstance(metrics, Iterable) and len(metrics) > 0
        assert all([len(metrics_pair) == 2 for metrics_pair in metrics])
        assert x_limits is None or (isinstance(x_limits, Iterable) and len(x_limits) == len(metrics))
        assert y_limits is None or (isinstance(y_limits, Iterable) and len(y_limits) == len(metrics))
        if scale_multipliers is not None:
            assert isinstance(scale_multipliers, Iterable) and len(scale_multipliers) == len(metrics)
            assert all([len(scale_multiplier) == 2 for scale_multiplier in scale_multipliers])

        for metric_i, (metric_x, metric_y) in enumerate(metrics):
            fig = plt.figure(figsize=figsize)

            x_means, x_se, x_labels, x_raw = [], [], [], []
            y_means, y_se, y_labels, y_raw = [], [], [], []
            matrix_label = []
            for mode_i, mode_data in enumerate(analysis_metrics_per_mode):
                mode, available_metrics_list, group_pair_id, group_pair_name = mode_data

                if mode == "Same-play" or mode == "same training run":
                    mode = self.SELF_PLAY_MODE
                elif mode == "Cross-play" or mode == "cross training run":
                    mode = self.CROSS_PLAY_MODE

                print("Evaluator mode:", mode)
                if group_pair_name is not None and all([name is not None for name in group_pair_name]) and show_groups:
                    print("Using group_pair_name:", group_pair_name)
                    label = f"{mode}: " + " vs ".join(group_pair_name)
                else:
                    label = mode
                label = label.replace('_', ' ')

                x, y = [], []
                assert len(available_metrics_list) > 0
                counter = 0
                random.shuffle(available_metrics_list)
                for available_metrics in available_metrics_list:
                    # print("available_metrics:", available_metrics)
                    x_point = available_metrics[metric_x][metric_mode]
                    y_point = available_metrics[metric_y][metric_mode]
                    if scale_multipliers is not None:
                        x_point *= scale_multipliers[metric_i][0]
                        y_point *= scale_multipliers[metric_i][1]
                    x.append(x_point)
                    y.append(y_point)
                    counter += 1
                    if plot_max_n_points is not None and counter >= plot_max_n_points:
                        break
                if save_matrix:
                    x_means.append(sum(x) / len(x))
                    x_se.append(np.array(x).std() / np.sqrt(len(x)))
                    x_labels.append(f"Metric:{metric_x}, Metric mode:{metric_mode}")
                    y_means.append(sum(y) / len(y))
                    y_se.append(np.array(y).std() / np.sqrt(len(y)))
                    y_labels.append(f"Metric:{metric_y}, Metric mode:{metric_mode}")
                    matrix_label.append(label)
                    x_raw.append(x)
                    y_raw.append(y)
                print("x, y", x, y)
                if jitter is not None:
                    x += np.random.normal(loc=0.0, scale=jitter, size=(len(x),))
                    y += np.random.normal(loc=0.0, scale=jitter, size=(len(y),))

                plt.plot(x, y, 'o', color=colors[mode_i], label=label, alpha=alpha, markersize=markersize)

            plt.legend(numpoints=1, frameon=frameon)
            if add_title:
                plt.title(f"Same-play vs Cross-play{title_sufix}")
            plt.xlabel(metric_x if xlabel is None else xlabel)
            plt.ylabel(metric_y if ylabel is None else ylabel)
            if x_limits is not None:
                plt.xlim(x_limits[metric_i])
            if y_limits is not None:
                plt.ylim(y_limits[metric_i])
            if show:
                plt.show()
            if save_fig:
                file_name = f'same_and_cross_play_{metric_y}_vs_{metric_x}.png'
                file_name = file_name.replace('/', '_')
                file_path = os.path.join(self.exp_parent_dir, file_name)
                print("save fig to", file_path)
                fig.savefig(file_path, dpi=fig.dpi)
            if save_matrix:
                file_name = f'same_and_cross_play_{metric_y}_vs_{metric_x}_matrix.json'
                file_name = file_name.replace('/', '_')
                file_path = os.path.join(self.exp_parent_dir, file_name)
                formated_data = {}
                for x_mean, x_std_err, x_lbl, y_mean, y_std_err, y_lbl, lbl, x, y in zip(
                        x_means, x_se, x_labels, y_means, y_se, y_labels, matrix_label, x_raw, y_raw):
                    formated_data[lbl] = {
                        x_lbl: {"mean": x_mean, "std_err": x_std_err, "raw_data": str(x)},
                        y_lbl: {"mean": y_mean, "std_err": y_std_err, "raw_data": str(y)}
                    }
                with open(file_path, 'w') as f:
                    json.dump(formated_data, f, indent=4, sort_keys=True)

    def _get_group_pair_id(self, result):
        checkpoints_groups_sorted = self._extract_checkpoints_used(result)
        checkpoints_groups_sorted_str = [str(e) for e in checkpoints_groups_sorted]
        unique_group_pair_id = ''.join(checkpoints_groups_sorted_str)
        return unique_group_pair_id

    def _get_group_pair_names(self, result):
        checkpoints_groups_sorted = self._extract_checkpoints_used(result)
        if self.group_names is not None:
            group_pair_names = [self.group_names[group_idx] for group_idx in checkpoints_groups_sorted]
        else:
            group_pair_names = None
        return group_pair_names

    def _extract_checkpoints_used(self, result):
        checkpoints_used = [result[policy_id]["checkpoint_i"] for policy_id in self.policies_to_load_from_checkpoint]
        checkpoints_groups = [self.checkpoint_list_group_idx[checkpoint_i] for checkpoint_i in checkpoints_used]
        checkpoints_groups_sorted = sorted(checkpoints_groups)
        return checkpoints_groups_sorted
