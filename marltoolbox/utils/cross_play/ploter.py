import logging
import random
import copy
import pandas as pd

from marltoolbox.utils.cross_play import evaluator
from marltoolbox.utils.cross_play.stats_summary import StatisticSummary
from marltoolbox.utils.plot import PlotHelper, PlotConfig

logger = logging.getLogger(__name__)


class SelfAndCrossPlayPlotter:
    def __init__(self):
        self.x_axis_metric = None
        self.y_axis_metric = None
        self.metric_mode = None
        self.stat_summary = None
        self.data_groups_per_mode = None

    def plot_results(
        self,
        exp_parent_dir: str,
        x_axis_metric: str,
        y_axis_metric: str,
        metrics_per_mode: list,
        plot_config: PlotConfig,
        metric_mode: str = "avg",
    ):
        self._reset(x_axis_metric, y_axis_metric, metric_mode)
        for metrics_for_one_evaluation_mode in metrics_per_mode:
            self._extract_performance_evaluation_points(
                metrics_for_one_evaluation_mode
            )
        stat_summary_filename_prefix = (
            plot_config.filename_prefix
            + evaluator.RESULTS_SUMMARY_FILENAME_PREFIX
        )
        self.stat_summary.save_summary(
            filename_prefix=stat_summary_filename_prefix,
            folder_dir=exp_parent_dir,
        )
        return self._plot_and_save_fig(plot_config, exp_parent_dir)

    def _reset(self, x_axis_metric, y_axis_metric, metric_mode):
        self.x_axis_metric = x_axis_metric
        self.y_axis_metric = y_axis_metric
        self.metric_mode = metric_mode
        self.stat_summary = StatisticSummary(
            self.x_axis_metric, self.y_axis_metric, self.metric_mode
        )
        self.data_groups_per_mode = {}

    def _extract_performance_evaluation_points(
        self, metrics_for_one_evaluation_mode
    ):
        (
            mode,
            available_metrics_list,
            group_pair_id,
            group_pair_name,
        ) = metrics_for_one_evaluation_mode

        label = self._get_label(mode, group_pair_name)
        x, y = self._extract_x_y_points(available_metrics_list)

        self.stat_summary.aggregate_stats_on_data_points(x, y, label)
        self.data_groups_per_mode[label] = self._format_as_df(x, y)
        print("x, y", x, y)

    def _get_label(self, mode, group_pair_name):

        print("Evaluator mode:", mode)
        if self._suffix_needed(group_pair_name):
            ordered_group_pair_name = self._order_group_names(group_pair_name)
            print(
                "Using ordered_group_pair_name:",
                ordered_group_pair_name,
                "from group_pair_name:",
                group_pair_name,
            )
            label = f"{mode}: " + " vs ".join(ordered_group_pair_name)
        else:
            label = mode
        label = label.replace("_", " ")
        print("label", label)
        return label

    def _suffix_needed(self, group_pair_name):
        if group_pair_name is None:
            return False
        return all(
            [name is not None for name in group_pair_name.values()]
        ) and all(group_pair_name.values())

    def _order_group_names(self, group_pair_name_original):
        group_pair_name = copy.deepcopy(group_pair_name_original)
        ordered_group_pair_name = []
        for metric in (self.x_axis_metric, self.y_axis_metric):
            for policy_id, one_group_name in group_pair_name.items():
                print(
                    "_order_group_names policy_id in metric", policy_id, metric
                )
                if policy_id in metric:
                    ordered_group_pair_name.append(one_group_name)
                    group_pair_name.pop(policy_id)
                    break
        assert len(group_pair_name.keys()) == 0, (
            "group_pair_name_original.keys() "
            f"{group_pair_name_original.keys()} not in the metrics provided: "
            "(self.x_axis_metric, self.y_axis_metric) "
            f"{(self.x_axis_metric, self.y_axis_metric)}"
        )
        return ordered_group_pair_name

    def _extract_x_y_points(self, available_metrics_list):
        x, y = [], []
        assert len(available_metrics_list) > 0
        random.shuffle(available_metrics_list)

        for available_metrics in available_metrics_list:
            if self.x_axis_metric in available_metrics.keys():
                x_point = available_metrics[self.x_axis_metric][
                    self.metric_mode
                ]
            else:
                x_point = 123456789
                from ray.util.debug import log_once

                msg = (
                    f"x_axis_metric {self.x_axis_metric}"
                    " not in available_metrics "
                    f"{available_metrics.keys()}"
                )
                if log_once(msg):
                    logger.warning(msg)

            if self.y_axis_metric in available_metrics.keys():
                y_point = available_metrics[self.y_axis_metric][
                    self.metric_mode
                ]
            else:
                y_point = 123456789
                msg = (
                    f"y_axis_metric {self.y_axis_metric}"
                    " not in available_metrics "
                    f"{available_metrics.keys()}"
                )
                if log_once(msg):
                    logger.warning(msg)
            x.append(x_point)
            y.append(y_point)
        return x, y

    def _format_as_df(self, x, y):
        group_df_dict = {
            "": [
                (one_x_point, one_y_point)
                for one_x_point, one_y_point in zip(x, y)
            ]
        }
        group_df = pd.DataFrame(group_df_dict)
        return group_df

    def _plot_and_save_fig(self, plot_config, exp_parent_dir):
        plot_helper = PlotHelper(plot_config)
        plot_helper.plot_cfg.save_dir_path = exp_parent_dir
        return plot_helper.plot_dots(self.data_groups_per_mode)
