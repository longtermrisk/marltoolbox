import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

COLORS = list(mcolors.TABLEAU_COLORS) + list(mcolors.XKCD_COLORS)
MARKERS = ["o", "s", "v", "^", "<", ">", "P", "X", "D", "*"]

LOWER_ENVELOPE_SUFFIX = "_lower_envelope"
UPPER_ENVELOPE_SUFFIX = "_upper_envelope"


class PlotConfig:
    def __init__(self, *,
                 xlabel: str = None,
                 ylabel: str = None,
                 display_legend: bool = True,
                 legend_fontsize: str = "small",
                 save_dir_path: str = None,
                 title: str = None,
                 xlim: str = None,
                 ylim: str = None,
                 colors=COLORS,
                 figsize: tuple = None,
                 plot_max_n_points: int = None,
                 markersize=None,
                 alpha: float = 1.0,
                 jitter: float = 0.0,
                 filename_prefix: str = "plot",
                 x_scale_multiplier: float = 1.0,
                 y_scale_multiplier: float = 1.0,
                 empty_markers=True,
                 background_area_coord=None,
                 markers=None):
        """
        Lot of the parameters follow the matplotlib.pyplot API.
        See https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot.
        """
        # Matplotlib.pyplot API
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.colors = colors
        self.figsize = figsize
        self.markersize = markersize
        self.alpha = alpha
        self.legend_fontsize = legend_fontsize

        # Custom parameters
        self.plot_max_n_points = plot_max_n_points
        self.display_legend = display_legend
        self.save_dir_path = save_dir_path
        self.filename_prefix = filename_prefix
        self.x_scale_multiplier = x_scale_multiplier
        self.y_scale_multiplier = y_scale_multiplier
        self.jitter = jitter
        self.empty_markers = empty_markers
        self.background_area_coord = background_area_coord
        self.markers = MARKERS if markers is None else markers


class PlotHelper:

    def __init__(self, plot_config: PlotConfig):
        self.plot_cfg = plot_config

    def plot_lines(self, data_groups: dict):
        """

        :param data_groups: dictionary containing pandas.DataFrame where
        each column is a line to plot (exeption of the columns used for the
        lower and upper bound of the envelop of another line).
        :return:
        """
        fig = self._init_plot()

        all_label_plotted = []
        for group_index, (group_id, group_df) in enumerate(
                data_groups.items()):
            new_labels_plotted = self._plot_lines_for_one_group(
                self.plot_cfg.colors[group_index], group_id, group_df)
            all_label_plotted.extend(new_labels_plotted)
        print("all_label_plotted", all_label_plotted)

        return self._finalize_plot(fig)

    def _init_plot(self):
        fig = plt.figure(figsize=self.plot_cfg.figsize)
        return fig

    def _plot_lines_for_one_group(self, group_color, group_id, group_df):
        label_plotted = []
        for col in group_df.columns:
            if col.endswith(LOWER_ENVELOPE_SUFFIX):
                pass
            elif col.endswith(UPPER_ENVELOPE_SUFFIX):
                self._plot_one_envelope(group_color, group_df, col)
            else:
                label = self._get_label(group_id, col)
                plt.plot(group_df.index * self.plot_cfg.x_scale_multiplier,
                         group_df[col] * self.plot_cfg.y_scale_multiplier,
                         label=label, color=group_color)
                label_plotted.append(label)
        return label_plotted

    def _plot_one_envelope(self, group_color, group_df, col):
        upper_envelope_col = col
        lower_envelope_col = col.replace(UPPER_ENVELOPE_SUFFIX,
                                         LOWER_ENVELOPE_SUFFIX)
        assert lower_envelope_col in group_df.columns
        plt.fill_between(group_df.index * self.plot_cfg.x_scale_multiplier,
                         group_df[
                             lower_envelope_col] * self.plot_cfg.y_scale_multiplier,
                         group_df[
                             upper_envelope_col] * self.plot_cfg.y_scale_multiplier,
                         color=group_color, alpha=0.2)

    def _get_label(self, group_id, col):
        if col:
            label = f"{group_id}: {col}"
        else:
            label = group_id
        return label

    def _finalize_plot(self, fig):
        if self.plot_cfg.display_legend:
            plt.legend(numpoints=1,
                       frameon=True,
                       fontsize=self.plot_cfg.legend_fontsize)
        if self.plot_cfg.xlabel is not None:
            plt.xlabel(self.plot_cfg.xlabel)
        if self.plot_cfg.ylabel is not None:
            plt.ylabel(self.plot_cfg.ylabel)
        if self.plot_cfg.title is not None:
            plt.title(self.plot_cfg.title)
        if self.plot_cfg.xlim is not None:
            plt.xlim(self.plot_cfg.xlim)
        if self.plot_cfg.ylim is not None:
            plt.ylim(self.plot_cfg.ylim)
        if self.plot_cfg.background_area_coord is not None:
            self._add_background_area()
        if self.plot_cfg.save_dir_path is not None:
            file_name = f'{self.plot_cfg.filename_prefix}_{self.plot_cfg.ylabel}_vs' \
                        f'_{self.plot_cfg.xlabel}.png'
            file_name = file_name.replace('/', '_')
            file_path = os.path.join(self.plot_cfg.save_dir_path, file_name)
            print("save fig to", file_path)
            fig.savefig(file_path, dpi=fig.dpi)
        else:
            file_path = None
        plt.close(fig)
        return file_path

    def plot_dots(self, data_groups: dict):
        """
        :param data_groups: dict of groups (same color and label prefix) containing a DataFrame containing (x,
        y) tuples. Each column in a group DataFrame has a different marker.
        """
        fig = self._init_plot()

        self.counter_labels = 0
        all_label_plotted = []
        for group_index, (group_id, group_df) in enumerate(
                data_groups.items()):
            new_labels_plotted = self._plot_dotes_for_one_group(
                self.plot_cfg.colors[group_index], group_id, group_df)
            all_label_plotted.extend(new_labels_plotted)
        print("all_label_plotted", all_label_plotted)

        return self._finalize_plot(fig)

    def _plot_dotes_for_one_group(self, group_color, group_id, group_df):
        label_plotted = []
        for col in group_df.columns:
            x, y = self._select_n_points_to_plot(group_df, col)
            x, y = self._add_jitter_to_points(x, y)
            x, y = self._apply_scale_multiplier(x, y)
            label = self._get_label(group_id, col)

            plt.plot(x, y,
                     markerfacecolor='none' if self.plot_cfg.empty_markers else group_color,
                     markeredgecolor=group_color, linestyle='None',
                     marker=self.plot_cfg.markers[self.counter_labels],
                     color=group_color, label=label,
                     alpha=self.plot_cfg.alpha,
                     markersize=self.plot_cfg.markersize)
            self.counter_labels += 1
            label_plotted.append(label)
        return label_plotted

    def _select_n_points_to_plot(self, group_df, col):
        if self.plot_cfg.plot_max_n_points is not None:
            n_points_to_plot = min(self.plot_cfg.plot_max_n_points,
                                   len(group_df))
            print(f"Selected {n_points_to_plot} n_points_to_plot")
        else:
            n_points_to_plot = len(group_df)
        group_df_sample = group_df.sample(n=int(n_points_to_plot))
        points = group_df_sample[col].tolist()
        x, y = [p[0] for p in points], [p[1] for p in points]
        return x, y

    def _add_jitter_to_points(self, x, y):
        if self.plot_cfg.jitter:
            x += np.random.normal(loc=0.0,
                                  scale=self.plot_cfg.jitter / self.plot_cfg.x_scale_multiplier,
                                  size=(len(x),))
            y += np.random.normal(loc=0.0,
                                  scale=self.plot_cfg.jitter / self.plot_cfg.y_scale_multiplier,
                                  size=(len(y),))
        return x, y

    def _apply_scale_multiplier(self, x, y):
        x_scaled = [x_p * self.plot_cfg.x_scale_multiplier for x_p in x]
        y_scaled = [y_p * self.plot_cfg.y_scale_multiplier for y_p in y]
        return x_scaled, y_scaled

    def _add_background_area(self):
        from scipy.spatial import ConvexHull
        assert self.plot_cfg.background_area_coord.ndim == 3
        points_defining_area = self.plot_cfg.background_area_coord.flatten().reshape(
            -1, 2)
        area_hull = ConvexHull(points_defining_area)
        plt.fill(points_defining_area[area_hull.vertices, 0],
                 points_defining_area[area_hull.vertices, 1],
                 facecolor='none', edgecolor='purple', linewidth=1)
        plt.fill(points_defining_area[area_hull.vertices, 0],
                 points_defining_area[area_hull.vertices, 1],
                 'purple', alpha=0.05)
