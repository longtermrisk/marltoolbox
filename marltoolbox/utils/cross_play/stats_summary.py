import json
import os

import numpy as np


class StatisticSummary:
    def __init__(self, x_axis_metric, y_axis_metric, metric_mode):
        self.x_means, self.x_se, self.x_labels, self.x_raw = [], [], [], []
        self.y_means, self.y_se, self.y_labels, self.y_raw = [], [], [], []
        self.matrix_label = []
        self.x_axis_metric, self.y_axis_metric = x_axis_metric, y_axis_metric
        self.metric_mode = metric_mode

    def aggregate_stats_on_data_points(self, x, y, label):
        # TODO refactor that to use a data structure
        #  (like per metric and per plot?)
        self.x_means.append(sum(x) / len(x))
        self.x_se.append(np.array(x).std() / np.sqrt(len(x)))
        self.x_labels.append(
            f"Metric:{self.x_axis_metric}, " f"Metric mode:{self.metric_mode}"
        )

        self.y_means.append(sum(y) / len(y))
        self.y_se.append(np.array(y).std() / np.sqrt(len(y)))
        self.y_labels.append(
            f"Metric:{self.y_axis_metric}, " f"Metric mode:{self.metric_mode}"
        )

        self.matrix_label.append(label)
        self.x_raw.append(x)
        self.y_raw.append(y)

    def save_summary(self, filename_prefix, folder_dir):
        file_name = (
            f"{filename_prefix}_{self.y_axis_metric}_"
            f"vs_{self.x_axis_metric}_matrix.json"
        )
        file_name = file_name.replace("/", "_")
        file_path = os.path.join(folder_dir, file_name)
        formated_data = {}
        for step_i in range(len(self.x_means)):
            (
                x_mean,
                x_std_err,
                x_lbl,
                y_mean,
                y_std_err,
                y_lbl,
                lbl,
                x,
                y,
            ) = self._get_values_from_a_data_point(step_i)
            formated_data[lbl] = {
                x_lbl: {
                    "mean": x_mean,
                    "std_err": x_std_err,
                    "raw_data": str(x),
                },
                y_lbl: {
                    "mean": y_mean,
                    "std_err": y_std_err,
                    "raw_data": str(y),
                },
            }
        with open(file_path, "w") as f:
            json.dump(formated_data, f, indent=4, sort_keys=True)

    def _get_values_from_a_data_point(self, step_i):
        return (
            self.x_means[step_i],
            self.x_se[step_i],
            self.x_labels[step_i],
            self.y_means[step_i],
            self.y_se[step_i],
            self.y_labels[step_i],
            self.matrix_label[step_i],
            self.x_raw[step_i],
            self.y_raw[step_i],
        )
