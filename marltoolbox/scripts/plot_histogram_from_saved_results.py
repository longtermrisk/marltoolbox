import json
import os
from collections import namedtuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("agg")
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"font.size": 12})

COLORS = list(mcolors.TABLEAU_COLORS) + list(mcolors.XKCD_COLORS)
RANDOM_MARKERS = ["1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+"]
MARKERS = ["o", "s", "v", "^", "<", ">", "P", "X", "D", "*"] + RANDOM_MARKERS

Exp_data = namedtuple("Exp_data", ["base_algo", "env", "perf"])
Perf = namedtuple("Perf", ["mean", "std_dev", "std_err", "raw"])
File_data = namedtuple(
    "File_data",
    [
        "base_algo",
        "env",
        "reward_adaptation_divider",
        "path_to_self_play",
        "path_to_preferences",
        "max_r_by_players",
        "min_r_by_players",
    ],
)

NA = "N/A"

PLAYER_0 = 0
PLAYER_1 = 1


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    perf_per_mode_per_files = []
    for file_paths, file_data in zip(files_to_process, files_data):
        perf_per_mode = _get_stats(file_paths, n_players, file_data)
        perf_per_mode_per_files.append(
            Exp_data(file_data.base_algo, file_data.env, perf_per_mode)
        )

    _plot_bars(perf_per_mode_per_files)


IPD_MAX = (-1, -1)
IPD_MIN = (-3, -3)
ASYMIBOS_MAX = ((4 + 2) / 2.0, (2 + 1) / 2.0)
ASYMIBOS_MIN = (0, 0)
N_CELLS_AT_1_STEP = 4
N_CELLS_AT_2_STEPS = 4
N_CELLS_EXCLUDING_CURRENT = 8
MAX_PICK_SPEED = (
    N_CELLS_AT_1_STEP / N_CELLS_EXCLUDING_CURRENT / 1
    + N_CELLS_AT_2_STEPS / N_CELLS_EXCLUDING_CURRENT / 2
)  # 0.75
CG_MAX = (1.0 * MAX_PICK_SPEED / 2.0, 1.0 * MAX_PICK_SPEED / 2.0)
CG_MIN = (0, 0)
MCPCG_MAX = (
    (2 / 2.0 + 1 / 2.0) * MAX_PICK_SPEED / 2.0,
    (3 / 2.0 + 1 / 2.0) * MAX_PICK_SPEED / 2.0,
)
MCPCG_MIN = (0, 0)

LOLA_EXACT_WT_IPD_IDX = 1
N_NO_MCP = 4


def _get_inputs():
    prefix = "~/dev-maxime/CLR/vm-data/"
    files_data = (
        File_data(
            "amTFT",
            "IPD",
            20.0,
            "instance-60-cpu-1"
            "-preemtible/amTFT/2021_05_11/07_31_41/eval/2021_05_11/09_20_44"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            "instance-60-cpu-1"
            "-preemtible/amTFT/2021_05_11/07_31_41/eval/2021_05_11/09_20_44"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            IPD_MAX,
            IPD_MIN,
        ),
        File_data(
            "LOLA-Exact",
            "IPD",
            200.0,
            "instance-60-cpu-1"
            "-preemtible/LOLA_Exact/2021_05_11/07_46_03/eval/2021_05_11"
            "/07_49_14"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            None,
            # "instance-60-cpu-1"
            # "-preemtible/LOLA_Exact/2021_05_11/07_46_03/eval/2021_05_11/07_49_14"
            # "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            IPD_MAX,
            IPD_MIN,
        ),
        File_data(
            "amTFT",
            "CG",
            100.0,
            "instance-60-cpu-3-preemtible"
            "/amTFT/2021_05_11/08_02_22/eval/2021_05_11/17_32_31"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            "instance-60-cpu-3-preemtible"
            "/amTFT/2021_05_11/08_02_22/eval/2021_05_11/17_32_31"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            CG_MAX,
            CG_MIN,
        ),
        File_data(
            "LOLA-PG",
            "CG",
            40.0,
            "instance-10-cpu-8-memory-x2"
            "/LOLA_PG/2021_05_12/10_09_50/eval/2021_05_12/21_38_50"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            None,
            # "instance-10-cpu-8-memory-x2"
            # "/LOLA_PG/2021_05_12/10_09_50/eval/2021_05_12/21_38_50"
            # "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            CG_MAX,
            CG_MIN,
        ),
        File_data(
            "amTFT",
            "AsymIBoS",
            20.0,
            "instance-60-cpu-1"
            "-preemtible/amTFT/2021_05_11/07_40_04/eval/2021_05_11/11_43_26"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            "instance-60-cpu-1"
            "-preemtible/amTFT/2021_05_11/07_40_04/eval/2021_05_11/11_43_26"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            ASYMIBOS_MAX,
            ASYMIBOS_MIN,
        ),
        File_data(
            "LOLA-Exact",
            "AsymIBoS",
            200.0,
            "instance-60-cpu-1"
            "-preemtible/LOLA_Exact/2021_05_11/07_47_16/eval/2021_05_11"
            "/07_50_36"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            "instance-60-cpu-1"
            "-preemtible/LOLA_Exact/2021_05_11/07_47_16/eval/2021_05_11/07_50_36"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            ASYMIBOS_MAX,
            ASYMIBOS_MIN,
        ),
        File_data(
            "amTFT",
            "MCPCG",
            100.0,
            "instance-10-cpu-7-memory-x2"
            "/amTFT/2021_05_09/08_20_23/eval/2021_05_09/23_21_35"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            "instance-10-cpu-7-memory-x2"
            "/amTFT/2021_05_09/08_20_23/eval/2021_05_09/23_21_35"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            MCPCG_MAX,
            MCPCG_MIN,
        ),
        File_data(
            "LOLA-PG",
            "MCPCG",
            40.0,
            "instance-60-cpu-3-preemtible"
            "/LOLA_PG/2021_05_09/15_54_03/eval/2021_05_09/23_12_34"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            "instance-60-cpu-3-preemtible"
            "/LOLA_PG/2021_05_09/15_54_03/eval/2021_05_09/23_12_34"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            MCPCG_MAX,
            MCPCG_MIN,
        ),
    )
    n_players = 2
    return prefix, files_data, n_players


def _preprocess_inputs(prefix, files_data):
    files_to_process = [
        (
            os.path.join(prefix, file_data.path_to_self_play),
            os.path.join(prefix, file_data.path_to_preferences)
            if file_data.path_to_preferences is not None
            else None,
        )
        for file_data in files_data
    ]
    return files_to_process


def _get_stats(file_paths, n_players, file_data):
    self_play_path = file_paths[0]
    perf_per_mode = _get_stats_for_file(self_play_path, n_players, file_data)
    print("perf_per_mode", perf_per_mode)
    self_play = perf_per_mode["self-play"]
    cross_play = perf_per_mode["cross-play"]

    preference_path = file_paths[1]
    if preference_path is not None:
        perf_per_mode_bis = _get_stats_for_file(
            preference_path, n_players, file_data
        )
        same_preferences_cross_play = perf_per_mode_bis[
            "cross-play: same pref vs same pref"
        ]
        if "cross-play: diff pref vs diff pref" in perf_per_mode_bis.keys():
            diff_preferences_cross_play = perf_per_mode_bis[
                "cross-play: diff pref vs diff pref"
            ]
        else:
            diff_preferences_cross_play = NA
    else:
        same_preferences_cross_play = NA
        diff_preferences_cross_play = NA

    all_perf = [
        self_play,
        cross_play,
        same_preferences_cross_play,
        diff_preferences_cross_play,
    ]
    return all_perf


def _get_stats_for_file(file, n_players, file_data):
    perf_per_mode = {}
    file_path = os.path.expanduser(file)
    with (open(file_path, "rb")) as f:
        file_content = json.load(f)
        for eval_mode, mode_perf in file_content.items():
            perf = [None] * 2
            print("eval_mode", eval_mode)
            for metric, metric_perf in mode_perf.items():
                player_idx = _extract_player_idx(metric)

                perf_per_replicat = np.array(
                    _convert_str_of_list_to_list(metric_perf["raw_data"])
                )

                n_replicates_in_content = len(perf_per_replicat)
                values_per_replicat_per_player = _adapt_values(
                    perf_per_replicat, file_data, player_idx
                )

                print(
                    "values_per_replicat_per_player",
                    values_per_replicat_per_player,
                )
                mean_per_player = values_per_replicat_per_player.mean(axis=0)
                std_dev_per_player = values_per_replicat_per_player.std(axis=0)
                std_err_per_player = std_dev_per_player / np.sqrt(
                    n_replicates_in_content
                )
                perf[player_idx] = Perf(
                    mean_per_player,
                    std_dev_per_player,
                    std_err_per_player,
                    values_per_replicat_per_player,
                )
            perf_per_mode[eval_mode] = perf

    return perf_per_mode


def _extract_player_idx(metric):
    if "player_row" in metric:
        player_idx = PLAYER_0
    elif "player_col" in metric:
        player_idx = PLAYER_1
    elif "player_red" in metric:
        player_idx = PLAYER_0
    elif "player_blue" in metric:
        player_idx = PLAYER_1
    else:
        raise ValueError()
    return player_idx


def _adapt_values(values_per_replicat_per_player, file_data, player_idx):
    scaled_values = (
        values_per_replicat_per_player / file_data.reward_adaptation_divider
    )
    normalized_values = scaled_values - file_data.min_r_by_players[player_idx]
    normalized_values = normalized_values / (
        file_data.max_r_by_players[player_idx]
        - file_data.min_r_by_players[player_idx]
    )
    return normalized_values


def _convert_str_of_list_to_list(str_of_list):
    return [
        float(v)
        for v in str_of_list.replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .split(",")
    ]


def _plot_bars(perf_per_mode_per_files):
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    _, x, groups = _plot_merged_players(perf_per_mode_per_files, mcp=False)
    plt.xticks(x, groups, rotation=15)
    plt.ylabel("Normalized scores")

    plt.subplot(122)
    legend, x, groups = _plot_merged_players(perf_per_mode_per_files, mcp=True)
    plt.xticks(x, groups, rotation=15)
    plt.ylabel("Normalized scores")

    plt.legend(
        legend,
        frameon=True,
        bbox_to_anchor=(1.0, -0.23),
    )

    # Save the figure and show
    plt.tight_layout(rect=[0, -0.05, 1.0, 1.0])
    plt.savefig("bar_plot.png")
    plt.show()


def _extract_value(all_perf, idx, player_idx, attrib):
    return [
        getattr(el[idx][player_idx], attrib)
        if hasattr(el[idx][player_idx], attrib)
        else 0.0
        for el in all_perf
    ]


def _plot_merged_players(perf_per_mode_per_files, mcp: bool):
    all_perf = [el.perf for el in perf_per_mode_per_files]
    groups = [f"{el.env}+{el.base_algo}" for el in perf_per_mode_per_files]

    width = 0.1
    x_delta = 0.3

    self_play_p0 = _extract_value(all_perf, 0, PLAYER_0, "raw")
    self_play_p1 = _extract_value(all_perf, 0, PLAYER_1, "raw")
    cross_play_p0 = _extract_value(all_perf, 1, PLAYER_0, "raw")
    cross_play_p1 = _extract_value(all_perf, 1, PLAYER_1, "raw")
    same_pref_p0 = _extract_value(all_perf, 2, PLAYER_0, "raw")
    same_pref_p1 = _extract_value(all_perf, 2, PLAYER_1, "raw")
    diff_pref_p0 = _extract_value(all_perf, 3, PLAYER_0, "raw")
    diff_pref_p1 = _extract_value(all_perf, 3, PLAYER_1, "raw")

    self_play = _avg_over_players(self_play_p0, self_play_p1)
    cross_play = _avg_over_players(cross_play_p0, cross_play_p1)
    same_pref_perf = _avg_over_players(same_pref_p0, same_pref_p1)
    diff_pref_perf = _avg_over_players(diff_pref_p0, diff_pref_p1)

    self_play_err = _get_std_err(self_play)
    cross_play_err = _get_std_err(cross_play)
    same_pref_perf_err = _get_std_err(same_pref_perf)
    diff_pref_perf_err = _get_std_err(diff_pref_perf)

    self_play = _get_mean(self_play)
    cross_play = _get_mean(cross_play)
    same_pref_perf = _get_mean(same_pref_perf)
    diff_pref_perf = _get_mean(diff_pref_perf)

    # We can't infer preferences (or pseudo-welfare function) in LOLA-Exact in
    # IPD since only one equilibrium is generated. Thus removing these badly
    # named data points
    # same_pref_perf[LOLA_EXACT_WT_IPD_IDX] = 0.0
    # diff_pref_perf[LOLA_EXACT_WT_IPD_IDX] = 0.0

    if not mcp:
        plt.text(1.33, 0.04, NA, fontdict={"fontsize": 10.0, "rotation": 90})
        plt.text(1.43, 0.04, NA, fontdict={"fontsize": 10.0, "rotation": 90})
        self_play = self_play[:N_NO_MCP]
        cross_play = cross_play[:N_NO_MCP]
        same_pref_perf = same_pref_perf[:N_NO_MCP]
        diff_pref_perf = diff_pref_perf[:N_NO_MCP]
        self_play_err = self_play_err[:N_NO_MCP]
        cross_play_err = cross_play_err[:N_NO_MCP]
        same_pref_perf_err = same_pref_perf_err[:N_NO_MCP]
        diff_pref_perf_err = diff_pref_perf_err[:N_NO_MCP]
        groups = groups[:N_NO_MCP]
        subplot_name = "Not MCPs"
    else:
        self_play = self_play[N_NO_MCP:]
        cross_play = cross_play[N_NO_MCP:]
        same_pref_perf = same_pref_perf[N_NO_MCP:]
        diff_pref_perf = diff_pref_perf[N_NO_MCP:]
        self_play_err = self_play_err[N_NO_MCP:]
        cross_play_err = cross_play_err[N_NO_MCP:]
        same_pref_perf_err = same_pref_perf_err[N_NO_MCP:]
        diff_pref_perf_err = diff_pref_perf_err[N_NO_MCP:]
        groups = groups[N_NO_MCP:]
        subplot_name = "MCPs"
    plt.text(1.0, 1.1, subplot_name, fontdict={"fontsize": 14.0})

    x = np.arange(len(self_play))

    plt.bar(
        x_delta + x - width * 1.5,
        self_play,
        width,
        yerr=self_play_err,
    )
    plt.bar(
        x_delta + x - width * 0.5,
        cross_play,
        width,
        yerr=cross_play_err,
    )
    plt.bar(
        x_delta + x + width * 0.5,
        same_pref_perf,
        width,
        yerr=same_pref_perf_err,
    )
    plt.bar(
        x_delta + x + width * 1.5,
        diff_pref_perf,
        width,
        yerr=diff_pref_perf_err,
    )
    legend = [
        "Self-play",
        "Cross-play",
        "Cross-play between identical preferences",
        "Cross-play between different preferences",
    ]

    return legend, x, groups


def _avg_over_players(values_player0, values_player1):
    return [
        (np.array(v_p0) + np.array(v_p1)) / 2
        for v_p0, v_p1 in zip(values_player0, values_player1)
    ]


def _get_std_err(values):
    return [
        v.std() / np.sqrt(v.shape[0]) if len(v.shape) > 0 else 0.0
        for v in values
    ]


def _get_mean(values):
    return [v.mean() if len(v.shape) > 0 else 0.0 for v in values]


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)