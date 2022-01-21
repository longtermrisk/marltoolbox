import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1e-6


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    for i, (file_path, file_data) in enumerate(
        zip(files_to_process, files_data)
    ):
        meta_policies, welfares = _get_policies(file_path)
        actions_names = _get_actions_names(welfares)
        plot_policies(
            meta_policies,
            actions_names,
            title=file_data[0],
            path_suffix=f"_{i}",
            announcement_protocol=True,
        )


def _get_inputs():
    prefix = "~/dev-maxime/CLR/vm-data/"
    files_data = (
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(alpha-rank mixed on welfare sets)  & BASE(announcement + "
            "LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14"
            "/10_37_24/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(alpha-rank pure on welfare sets)  & BASE(announcement + "
            "LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14/10_39_47/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(replicator dynamic random init on welfare sets)  & BASE("
            "announcement + LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14/10_42_10/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(replicator dynamic default init on welfare sets)  & BASE("
            "announcement + LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14/10_46_23/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(baseline random)  & BASE(" "announcement + LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14"
            "/10_50_36/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(PG)  & BASE(" "announcement + LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14/10_52_43/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(LOLA-Exact)  & BASE(" "announcement + LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14"
            "/11_00_02/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(SOS-Exact)  & BASE(" "announcement + LOLA-Exact)",
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_14"
            "/12_38_59/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
    )
    n_players = 2
    return prefix, files_data, n_players


def _preprocess_inputs(prefix, files_data):
    files_to_process = [
        os.path.expanduser(os.path.join(prefix, file_data[1]))
        for file_data in files_data
    ]
    return files_to_process


def _get_policies(file_path):
    parent_dir, _ = os.path.split(file_path)
    parent_parent_dir, _ = os.path.split(parent_dir)
    meta_policies_file = os.path.join(parent_parent_dir, "meta_policies.json")
    with (open(meta_policies_file, "rb")) as f:
        meta_policies = json.load(f)
    meta_policies = meta_policies["meta_policies"]
    print("meta_policies", type(meta_policies), meta_policies)

    parent_parent_parent_dir, _ = os.path.split(parent_parent_dir)
    welfares_file = os.path.join(
        parent_parent_parent_dir, "payoffs_matrices_0.json"
    )
    with (open(welfares_file, "rb")) as f:
        welfares = json.load(f)
    welfares = welfares["welfare_fn_sets"]
    print("welfares", type(welfares), welfares)

    return meta_policies, welfares


def _get_actions_names(welfares):
    actions_names = (
        welfares.replace("OrderedSet", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .lstrip()
    )
    actions_names = actions_names.split("],")
    actions_names = [f"({el})" for el in actions_names]
    actions_names = [
        el.replace("]", "").replace(" ", "") for el in actions_names
    ]
    return actions_names


def plot_policies(
    meta_policies,
    actions_names,
    title=None,
    path_prefix="",
    path_suffix="",
    announcement_protocol=False,
):
    """
    Given the policies of two players, create plots to vizualize them.

    :param meta_policies:
    :param actions_names: names of actions of each players
    :param title: title of plot
    :param path_prefix: prefix to plot save path
    :param path_suffix: suffix to plot save path
    :return:
    """
    plt.style.use("default")

    if len(actions_names) != 2:
        actions_names = (actions_names, actions_names)
    print("actions_names", actions_names, "len", len(actions_names[0]))
    policies_p0 = []
    policies_p1 = []
    for meta_policy in meta_policies:
        policies_p0.append(meta_policy["player_row"])
        policies_p1.append(meta_policy["player_col"])
    print("policies_p0", len(policies_p0), "policies_p1", len(policies_p1))
    policies_p0 = np.array(policies_p0)
    policies_p1 = np.array(policies_p1)

    if announcement_protocol:
        fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
    else:
        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16))

    if title is not None:
        fig.suptitle(
            title,
            fontweight="bold",
        )
    _plot_means(policies_p0, policies_p1, actions_names, ax)
    _plot_std_dev(policies_p0, policies_p1, actions_names, ax2)
    _plot_joint_policies_vanilla(
        policies_p0,
        policies_p1,
        actions_names,
        ax3,
    )
    if announcement_protocol:
        _plot_joint_policies_wt_announcement_protocol(
            policies_p0, policies_p1, actions_names, ax4
        )

    plt.tight_layout()
    path_prefix = os.path.expanduser(path_prefix)
    plt.savefig(f"{path_prefix}meta_policies{path_suffix}.png")

    plt.style.use("seaborn-whitegrid")


def _plot_means(policies_p0, policies_p1, actions_names, ax):
    policies_p0_mean = policies_p0.mean(axis=0)
    policies_p1_mean = policies_p1.mean(axis=0)
    policies_mean = np.stack([policies_p0_mean, policies_p1_mean], axis=0)
    im, cbar = heatmap(
        policies_mean,
        ["player_row", "player_col"],
        actions_names[0],
        ax=ax,
        cmap="YlGn",
        cbarlabel="MEAN proba",
    )
    texts = annotate_heatmap(im, valfmt="{x:.3f}")


def _plot_std_dev(policies_p0, policies_p1, actions_names, ax):
    policies_p0_std = policies_p0.std(axis=0)
    policies_p1_std = policies_p1.std(axis=0)
    policies_std = np.stack([policies_p0_std, policies_p1_std], axis=0)
    im, cbar = heatmap(
        policies_std,
        ["player_row", "player_col"],
        actions_names[1],
        ax=ax,
        cmap="YlGn",
        cbarlabel="STD",
    )
    texts = annotate_heatmap(im, valfmt="{x:.3f}")


def _plot_joint_policies_vanilla(policies_p0, policies_p1, actions_names, ax):
    policies_p0 = np.expand_dims(policies_p0, axis=-1)
    policies_p1 = np.expand_dims(policies_p1, axis=-1)
    policies_p1 = np.transpose(policies_p1, (0, 2, 1))
    joint_policies = np.matmul(policies_p0, policies_p1)
    # print("joint_policies", joint_policies[0])
    _plot_joint_policies(joint_policies, actions_names, ax)


def _plot_joint_policies_wt_announcement_protocol(
    policies_p0, policies_p1, actions_names, ax
):
    joint_policies, actions_names = _preprocess_announcement_protocol(
        policies_p0, policies_p1, actions_names
    )
    _plot_joint_policies(joint_policies, actions_names, ax)


def _plot_joint_policies(joint_policies, actions_names, ax):
    assert np.all(
        np.abs(joint_policies.sum(axis=2).sum(axis=1) - 1.0) < EPSILON
    ), f"{np.abs(joint_policies.sum(axis=2).sum(axis=1) - 1.0)}"
    joint_policies = joint_policies.mean(axis=0)

    im, cbar = heatmap(
        joint_policies,
        actions_names[0],
        actions_names[1],
        ax=ax,
        cmap="YlGn",
        cbarlabel="Joint policy",
    )
    texts = annotate_heatmap(im, valfmt="{x:.3f}")


def _preprocess_announcement_protocol(policies_p0, policies_p1, actions_names):
    welfare_sets_announced, welfares = _convert_to_welfare_sets(actions_names)
    n_welfare_fn = len(welfares)
    n_replicates = len(policies_p0)
    joint_policies = np.zeros(shape=(n_replicates, n_welfare_fn, n_welfare_fn))
    for relpi_i, (pi_pl0_repl, pi_pl1_repl) in enumerate(
        zip(policies_p0, policies_p1)
    ):
        for w_set_pl0, pi_0 in zip(welfare_sets_announced[0], pi_pl0_repl):
            for w_set_pl1, pi_1 in zip(welfare_sets_announced[1], pi_pl1_repl):
                intersection = w_set_pl0 & w_set_pl1
                n_welfare_fn_in_intersec = len(intersection)
                if n_welfare_fn_in_intersec > 0:
                    for welfare in intersection:
                        welfare_idx = welfares.index(welfare)
                        joint_policies[relpi_i, welfare_idx, welfare_idx] += (
                            pi_0 * pi_1
                        ) / n_welfare_fn_in_intersec
                else:
                    welfare_idx_pl0 = welfares.index("utilitarian")
                    welfare_idx_pl1 = welfares.index("egalitarian")
                    joint_policies[
                        relpi_i, welfare_idx_pl0, welfare_idx_pl1
                    ] += (pi_0 * pi_1)

    return joint_policies, [welfares, welfares]


def _convert_to_welfare_sets(actions_names):
    welfare_sets_announced = []
    all_wefares = []
    for actions_names_of_player_i in actions_names:
        actions_names_of_player_i = [
            str(el) for el in actions_names_of_player_i
        ]
        sets_player_i = []
        for action_name in actions_names_of_player_i:
            welfare_set = (
                action_name.replace("'", "")
                .replace("(", "")
                .replace(")", "")
                .replace("OrderedSet", "")
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
                .split("," "")
            )
            sets_player_i.append(set(welfare_set))
            all_wefares.extend(welfare_set)
        welfare_sets_announced.append(sets_player_i)
    all_wefares = tuple(sorted(tuple(set(all_wefares))))
    assert len(all_wefares) <= 3, f"{all_wefares}"
    return welfare_sets_announced, all_wefares


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor"
    )

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)
    for k, v in ax.spines.items():
        ax.spines[k].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar

    # return im, None


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
