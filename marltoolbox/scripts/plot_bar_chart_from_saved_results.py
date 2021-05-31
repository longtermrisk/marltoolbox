import json
import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("agg")
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"font.size": 12})

COLORS = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

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
        "welfare_optim",
        "welfare_functions",
    ],
)
Final_values = namedtuple(
    "Final_values",
    [
        "base_algo",
        "env",
        "self_play",
        "cross_play",
        "cross_play_same",
        "cross_play_diff",
    ],
)
NA = "N/A"

PLAYER_0 = 0
PLAYER_1 = 1

VALUE_TERM = "value"
SPLIT_NEGO = False
REMOVE_STARS = True

USING_METRIC = 1
# USING_METRIC = 2
# USING_METRIC = 3
# USING_METRIC = 4
ABCG_USE_MEAN_DISAGREEMENT = None
# metric (1) Normalize(mean of ideals given welfare functions)
#   NORMALIZE_USE_EMPIRICAL_MAX = False & USE_DISTANCE_TO_WELFARE_OPTIM = False
# metric (2) Normalize(empirical max and ideal min)
#   NORMALIZE_USE_EMPIRICAL_MAX = True & USE_DISTANCE_TO_WELFARE_OPTIM = False
# metric (3) Min distance to the welfare-optimal profiles
#   NORMALIZE_USE_EMPIRICAL_MAX = False & USE_DISTANCE_TO_WELFARE_OPTIM = True
if USING_METRIC == 1:
    NORMALIZE_USE_EMPIRICAL_MAX = False
    USE_DISTANCE_TO_WELFARE_OPTIM = False
elif USING_METRIC == 2:
    NORMALIZE_USE_EMPIRICAL_MAX = True
    USE_DISTANCE_TO_WELFARE_OPTIM = False
elif USING_METRIC == 3:
    NORMALIZE_USE_EMPIRICAL_MAX = False
    USE_DISTANCE_TO_WELFARE_OPTIM = True
    ABCG_USE_MEAN_DISAGREEMENT = False
elif USING_METRIC == 4:
    NORMALIZE_USE_EMPIRICAL_MAX = False
    USE_DISTANCE_TO_WELFARE_OPTIM = False

if USE_DISTANCE_TO_WELFARE_OPTIM:
    NORMALIZED_NAME = f"Distance"
    assert not NORMALIZE_USE_EMPIRICAL_MAX
else:
    NORMALIZED_NAME = f"Normalized {VALUE_TERM}"

LEGEND = [
    "Self-play",
    "Cross-play, identical welfare functions",
    "Cross-play, different welfare functions",
]

LEGEND_NO_SPLIT = [
    "Self-play",
    "Cross-play",
]

GLOBAL_CROSS_PLAY_IN_LOLA = False
if GLOBAL_CROSS_PLAY_IN_LOLA:
    LEGEND = [
        "Self-play",
        "Cross-play",
        "Cross-play, identical welfare functions",
        "Cross-play, different welfare functions",
    ]
# LOLA_EXACT_WT_IPD_IDX = 1
N_NO_MCP = 4
NEGOTIATION_RATIO = 0.66

CG_N_STEPS = 100.0
ABCG_N_STEPS = 100.0
if USE_DISTANCE_TO_WELFARE_OPTIM:
    IPD_MAX = None
    IPD_MIN = None
    ASYMIBOS_MAX = None
    ASYMIBOS_MIN = None
    CG_MAX = None
    CG_MIN = None
    ABCG_MAX = None
    ABCG_MIN = None
    EMPIRICAL_WELFARE_OPTIMUM_CG = (
        (34.9875 / CG_N_STEPS, 34.89375 / CG_N_STEPS),
        (34.14375 / CG_N_STEPS, 34.05625 / CG_N_STEPS),
    )
    EMPIRICAL_WELFARE_OPTIMUM_ABCG = (
        (14.9640625 / ABCG_N_STEPS, 112.25625 / ABCG_N_STEPS),
        (34.41875 / ABCG_N_STEPS, 34.60625 / ABCG_N_STEPS),
    )
    ABCG_MEAN_DISAGREEMENT = (
        (17.886574074074073 + 2.6090686274509802) / (2 * ABCG_N_STEPS),
        (18.39814814814815 + 11.79656862745098) / (2 * ABCG_N_STEPS),
    )
else:
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
    ABCG_MAX = (
        (2 / 2.0 + 1 / 2.0) * MAX_PICK_SPEED / 2.0,
        (3 / 2.0 + 1 / 2.0) * MAX_PICK_SPEED / 2.0,
    )
    ABCG_MIN = (0, 0)
    EMPIRICAL_WELFARE_OPTIMUM_CG = None
    EMPIRICAL_WELFARE_OPTIMUM_ABCG = None
    ABCG_USE_MEAN_DISAGREEMENT = None
    ABCG_MEAN_DISAGREEMENT = None

UTILITARIAN_W = lambda xy: xy.sum(axis=1)
EGALITARIAN_W = lambda xy: xy.min(axis=1)
NASH_W = lambda xy: xy[:, 0] * xy[:, 1]

if USING_METRIC == 1:
    # (Mean, Std_err)
    negotiation_self_play = (
        0.4567 / NEGOTIATION_RATIO,
        0.0003 / NEGOTIATION_RATIO,
    )
    negotiation_cross_play = (
        0.4272 / NEGOTIATION_RATIO,
        0.0002 / NEGOTIATION_RATIO,
    )
    negotiation_same_play = (
        0.4526 / NEGOTIATION_RATIO,
        0.0003 / NEGOTIATION_RATIO,
    )
    negotiation_diff_play = (
        0.4017 / NEGOTIATION_RATIO,
        0.0004 / NEGOTIATION_RATIO,
    )
elif USING_METRIC == 2:
    # (Mean, Std_err)
    raise ValueError("not value for NEGOTIATION with mertic 2")
elif USING_METRIC == 3:
    # (Mean, Std_err)
    negotiation_self_play = (0.4014471264303261, 0.00041116296647391)
    negotiation_cross_play = (0.4673496367353227, 0.00030417354727912266)
    negotiation_same_play = (0.4041803573397903, 0.0004165843958125686)
    negotiation_diff_play = (0.5305189161308548, 0.000424625506684431)
elif USING_METRIC == 4:
    # (Mean, Std_err)
    # TODO fill Negotiation values for metric 4 here
    # negotiation_self_play = (..., ...)
    # negotiation_cross_play = (..., ...)
    # negotiation_same_play = (..., ...)
    # negotiation_diff_play = (..., ...)
    pass
else:
    raise ValueError()


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    perf_per_mode_per_files = []
    for file_paths, file_data in zip(files_to_process, files_data):
        perf_per_mode = _get_stats(file_paths, n_players, file_data)
        perf_per_mode_per_files.append(
            Exp_data(file_data.base_algo, file_data.env, perf_per_mode)
        )

    # Plot with 1 subplot
    # _plot_bars(perf_per_mode_per_files, welfare_split=True)
    # _plot_bars(perf_per_mode_per_files, welfare_split=False)

    # Plot with 2 subplots
    _plot_bars_separate(perf_per_mode_per_files, welfare_split=True)
    _plot_bars_separate(perf_per_mode_per_files, welfare_split=False)


def _get_inputs():
    # Files on Maxime's local machine
    prefix = "~/dev-maxime/CLR/vm-data/"
    # Files in unzipped folder
    # prefix = "./data/"
    files_data = (
        File_data(
            "amTFT",
            "IPD",
            20.0,
            "instance-60-cpu-1-preemtible"
            "/amTFT/2021_05_11/07_31_41/eval/2021_05_11/09_20_44"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            "instance-60-cpu-1-preemtible"
            "/amTFT/2021_05_11/07_31_41/eval/2021_05_11/09_20_44"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            IPD_MAX,
            IPD_MIN,
            (
                ((-1.0, -1.0),),
                (-3.0, -3.0),
            ),
            (UTILITARIAN_W, EGALITARIAN_W),
        ),
        File_data(
            "LOLA-Exact *",
            "IPD",
            200.0,
            "instance-60-cpu-1-preemtible"
            "/LOLA_Exact/2021_05_11/07_46_03/eval/2021_05_11"
            "/07_49_14"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            None,
            IPD_MAX,
            IPD_MIN,
            (
                ((-1.0, -1.0),),
                (-3.0, -3.0),
            ),
            (UTILITARIAN_W, EGALITARIAN_W),
        ),
        File_data(
            "amTFT",
            "CG",
            CG_N_STEPS,
            "instance-20-cpu-1-memory-x2"
            "/amTFT/2021_05_15/07_16_37/eval/2021_05_16/08_13_25"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            "instance-20-cpu-1-memory-x2"
            "/amTFT/2021_05_15/07_16_37/eval/2021_05_16/08_13_25"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            CG_MAX,
            CG_MIN,
            (EMPIRICAL_WELFARE_OPTIMUM_CG, (0.0, 0.0)),
            (UTILITARIAN_W, EGALITARIAN_W),
        ),
        File_data(
            "LOLA-PG *",
            "CG",
            40.0,
            "instance-20-cpu-1-memory-x2"
            "/LOLA_PG/2021_05_19/12_18_06/eval/2021_05_20/10_27_09"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            None,
            CG_MAX,
            CG_MIN,
            (EMPIRICAL_WELFARE_OPTIMUM_CG, (0.0, 0.0)),
            (UTILITARIAN_W, EGALITARIAN_W),
        ),
        File_data(
            "amTFT",
            "IAsymBoS",
            20.0,
            "instance-60-cpu-1-preemtible"
            "/amTFT/2021_05_11/07_40_04/eval/2021_05_11/11_43_26"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            "instance-60-cpu-1-preemtible"
            "/amTFT/2021_05_11/07_40_04/eval/2021_05_11/11_43_26"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            ASYMIBOS_MAX,
            ASYMIBOS_MIN,
            (((4.0, 1.0), (2.0, 2.0)), (0.0, 0.0)),
            (UTILITARIAN_W, EGALITARIAN_W),
        ),
        File_data(
            "LOLA-Exact",
            "IAsymBoS",
            200.0,
            "instance-60-cpu-1-preemtible"
            "/LOLA_Exact/2021_05_11/07_47_16/eval/2021_05_11"
            "/07_50_36"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            "instance-60-cpu-1-preemtible"
            "/LOLA_Exact/2021_05_11/07_47_16/eval/2021_05_11/07_50_36"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
            ASYMIBOS_MAX,
            ASYMIBOS_MIN,
            (((4.0, 1.0), (3.0, 1.5), (2.0, 2.0)), (0.0, 0.0)),
            (UTILITARIAN_W, EGALITARIAN_W, NASH_W),
        ),
        File_data(
            "amTFT",
            "ABCG",
            ABCG_N_STEPS,
            "instance-10-cpu-2"
            "/amTFT/2021_05_17/18_08_40/eval/2021_05_20/04_51_12"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            "instance-10-cpu-2"
            "/amTFT/2021_05_17/18_08_40/eval/2021_05_20/04_51_12"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            ABCG_MAX,
            ABCG_MIN,
            (
                EMPIRICAL_WELFARE_OPTIMUM_ABCG,
                ABCG_MEAN_DISAGREEMENT
                if ABCG_USE_MEAN_DISAGREEMENT
                else (0.0, 0.0),
            ),
            (UTILITARIAN_W, EGALITARIAN_W),
        ),
        File_data(
            "LOLA-PG **",
            "ABCG",
            40.0,
            "instance-60-cpu-2-preemtible"
            "/LOLA_PG/2021_05_19/08_17_37/eval/2021_05_19/18_02_21"
            "/plot_self_crossself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            "instance-60-cpu-2-preemtible"
            "/LOLA_PG/2021_05_19/08_17_37/eval/2021_05_19/18_02_21"
            "/plot_same_and_diff_prefself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json",
            ABCG_MAX,
            ABCG_MIN,
            (
                EMPIRICAL_WELFARE_OPTIMUM_ABCG,
                ABCG_MEAN_DISAGREEMENT
                if ABCG_USE_MEAN_DISAGREEMENT
                else (0.0, 0.0),
            ),
            (UTILITARIAN_W, EGALITARIAN_W),
        ),
        Final_values(
            "Negotiation",
            "REINFORCE-PS",
            # "self-play",
            (0.4567 / NEGOTIATION_RATIO, 0.0003 / NEGOTIATION_RATIO)
            if not USE_DISTANCE_TO_WELFARE_OPTIM
            else (0.4014471264303261, 0.00041116296647391),
            # "cross-play",
            (0.4272 / NEGOTIATION_RATIO, 0.0002 / NEGOTIATION_RATIO)
            if not USE_DISTANCE_TO_WELFARE_OPTIM
            else (0.4673496367353227, 0.00030417354727912266),
            # "cross-play same",
            (0.4526 / NEGOTIATION_RATIO, 0.0003 / NEGOTIATION_RATIO)
            if not USE_DISTANCE_TO_WELFARE_OPTIM
            else (0.4041803573397903, 0.0004165843958125686),
            # "cross-play diff",
            (0.4017 / NEGOTIATION_RATIO, 0.0004 / NEGOTIATION_RATIO)
            if not USE_DISTANCE_TO_WELFARE_OPTIM
            else (0.5305189161308548, 0.000424625506684431),
        ),
    )
    n_players = 2
    return prefix, files_data, n_players


def _preprocess_inputs(prefix, files_data):
    files_to_process = []
    for file_data in files_data:
        if isinstance(file_data, Final_values):
            value = file_data
        elif file_data.path_to_preferences is not None:
            value = (
                os.path.join(prefix, file_data.path_to_self_play),
                os.path.join(prefix, file_data.path_to_preferences),
            )
        else:
            value = (
                os.path.join(prefix, file_data.path_to_self_play),
                None,
            )
        files_to_process.append(value)

    return files_to_process


def _get_stats(file_paths, n_players, file_data):
    if isinstance(file_paths, Final_values):
        all_perf = file_paths
    else:
        self_play_path = file_paths[0]
        perf_per_mode = _get_stats_for_file(
            self_play_path, n_players, file_data
        )
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
            if (
                "cross-play: diff pref vs diff pref"
                in perf_per_mode_bis.keys()
            ):
                diff_preferences_cross_play = perf_per_mode_bis[
                    "cross-play: diff pref vs diff pref"
                ]
            else:
                diff_preferences_cross_play = NA
        else:
            same_preferences_cross_play = NA
            diff_preferences_cross_play = NA

        if NORMALIZE_USE_EMPIRICAL_MAX:
            (
                self_play,
                cross_play,
                same_preferences_cross_play,
                diff_preferences_cross_play,
            ) = _empirical_normalization(
                self_play,
                cross_play,
                same_preferences_cross_play,
                diff_preferences_cross_play,
                file_data,
            )

        if USING_METRIC == 4:
            (
                self_play,
                cross_play,
                same_preferences_cross_play,
                diff_preferences_cross_play,
            ) = _convert_to_metric_4(
                self_play,
                cross_play,
                same_preferences_cross_play,
                diff_preferences_cross_play,
                file_data,
            )

        all_perf = [
            self_play,
            cross_play,
            same_preferences_cross_play,
            diff_preferences_cross_play,
        ]
    return all_perf


def _convert_to_metric_4(
    self_play,
    cross_play,
    same_preferences_cross_play,
    diff_preferences_cross_play,
    file_data,
):
    all_values = []
    self_play_pair = np.stack(
        [self_play[PLAYER_0].raw, self_play[PLAYER_1].raw], axis=1
    )
    all_values.append(self_play_pair)
    cross_play_pair = np.stack(
        [cross_play[PLAYER_0].raw, cross_play[PLAYER_1].raw], axis=1
    )
    all_values.append(cross_play_pair)
    print("self_play_pair", self_play_pair.shape)
    print("cross_play_pair", cross_play_pair.shape)
    if not isinstance(same_preferences_cross_play, str):
        same_play_pair = np.stack(
            [
                same_preferences_cross_play[PLAYER_0].raw,
                same_preferences_cross_play[PLAYER_1].raw,
            ],
            axis=1,
        )
        all_values.append(same_play_pair)
        print("same_play_pair", same_play_pair.shape)
    if not isinstance(diff_preferences_cross_play, str):
        diff_play_pair = np.stack(
            [
                diff_preferences_cross_play[PLAYER_0].raw,
                diff_preferences_cross_play[PLAYER_1].raw,
            ],
            axis=1,
        )
        all_values.append(diff_play_pair)
        print("diff_play_pair", diff_play_pair.shape)

    all_values = np.concatenate(all_values, axis=0)
    print("all_values", all_values.shape)

    for k in [PLAYER_0, PLAYER_1]:
        self_play[k] = Perf(
            self_play[k].mean,
            self_play[k].std_dev,
            self_play[k].std_err,
            _compute_metric_4(
                self_play_pair,
                all_values,
                file_data,
            ),
        )
        cross_play[k] = Perf(
            cross_play[k].mean,
            cross_play[k].std_dev,
            cross_play[k].std_err,
            _compute_metric_4(
                cross_play_pair,
                all_values,
                file_data,
            ),
        )
        if not isinstance(same_preferences_cross_play, str):
            same_preferences_cross_play[k] = Perf(
                same_preferences_cross_play[k].mean,
                same_preferences_cross_play[k].std_dev,
                same_preferences_cross_play[k].std_err,
                _compute_metric_4(
                    same_play_pair,
                    all_values,
                    file_data,
                ),
            )
        if not isinstance(diff_preferences_cross_play, str):
            diff_preferences_cross_play[k] = Perf(
                diff_preferences_cross_play[k].mean,
                diff_preferences_cross_play[k].std_dev,
                diff_preferences_cross_play[k].std_err,
                _compute_metric_4(
                    diff_play_pair,
                    all_values,
                    file_data,
                ),
            )
    return (
        self_play,
        cross_play,
        same_preferences_cross_play,
        diff_preferences_cross_play,
    )


def _empirical_normalization(
    self_play,
    cross_play,
    same_preferences_cross_play,
    diff_preferences_cross_play,
    file_data,
):
    for k in [PLAYER_0, PLAYER_1]:
        possible_max = []
        possible_max.extend(self_play[k].raw)
        possible_max.extend(cross_play[k].raw)
        if not isinstance(same_preferences_cross_play, str):
            possible_max.extend(same_preferences_cross_play[k].raw)
        if not isinstance(diff_preferences_cross_play, str):
            possible_max.extend(diff_preferences_cross_play[k].raw)
        max_ = max(possible_max)
        min_ = file_data.min_r_by_players[k]
        self_play[k] = Perf(
            self_play[k].mean,
            self_play[k].std_dev,
            self_play[k].std_err,
            _normalize(self_play[k].raw, max_, min_),
        )
        cross_play[k] = Perf(
            cross_play[k].mean,
            cross_play[k].std_dev,
            cross_play[k].std_err,
            _normalize(cross_play[k].raw, max_, min_),
        )
        if not isinstance(same_preferences_cross_play, str):
            same_preferences_cross_play[k] = Perf(
                same_preferences_cross_play[k].mean,
                same_preferences_cross_play[k].std_dev,
                same_preferences_cross_play[k].std_err,
                _normalize(same_preferences_cross_play[k].raw, max_, min_),
            )
        if not isinstance(diff_preferences_cross_play, str):
            diff_preferences_cross_play[k] = Perf(
                diff_preferences_cross_play[k].mean,
                diff_preferences_cross_play[k].std_dev,
                diff_preferences_cross_play[k].std_err,
                _normalize(diff_preferences_cross_play[k].raw, max_, min_),
            )
    return (
        self_play,
        cross_play,
        same_preferences_cross_play,
        diff_preferences_cross_play,
    )


def _normalize(values, max_, min_):
    values = values - min_
    values = values / (max_ - min_)
    return values


def _get_stats_for_file(file, n_players, file_data):
    perf_per_mode = {}
    file_path = os.path.expanduser(file)
    with (open(file_path, "rb")) as f:
        file_content = json.load(f)
        possible_max = []
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

            if USE_DISTANCE_TO_WELFARE_OPTIM:
                metric = _distance_to_welfare_optimal_profiles(
                    np.stack([perf[PLAYER_0].raw, perf[PLAYER_1].raw], axis=1),
                    file_data,
                )

                perf[0] = Perf(
                    perf[0].mean,
                    perf[0].std_dev,
                    perf[0].std_err,
                    metric,
                )
                perf[1] = Perf(
                    perf[1].mean,
                    perf[1].std_dev,
                    perf[1].std_err,
                    metric,
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

    if not USE_DISTANCE_TO_WELFARE_OPTIM:
        if not NORMALIZE_USE_EMPIRICAL_MAX:
            if USING_METRIC != 4:
                assert USING_METRIC == 1
                normalized_values = _normalize(
                    scaled_values,
                    file_data.max_r_by_players[player_idx],
                    file_data.min_r_by_players[player_idx],
                )
                return normalized_values
    return scaled_values


def _distance_to_welfare_optimal_profiles(scaled_values, file_data):
    welfare_optimal_profiles = file_data.welfare_optim[0]
    disagrement_profile = file_data.welfare_optim[1]
    disagreement_to_optim_wel_prof = []
    for one_welfare_optimal_profile in welfare_optimal_profiles:
        # print("one_welfare_optimal_profile", one_welfare_optimal_profile)
        dist_disagr = np.linalg.norm(
            np.array(disagrement_profile)
            - np.array(one_welfare_optimal_profile)
        )
        # print("dist_disagr", dist_disagr)
        disagreement_to_optim_wel_prof.append(dist_disagr)
    disagreement_to_optim_wel_prof = min(disagreement_to_optim_wel_prof)

    # "numerator = min{ ||(u1, u2) - (4, 1)||, ||(u1, u2) - (3, 1.5)||, ||(u1, u2) - (2, 2)|| }
    # "denominator = min{ ||(0, 0) - (4, 1)||, ||(0, 0) - (3, 1.5)||, ||(0, 0) - (2, 2)|| }"

    scaled_values = np.array(scaled_values)
    print("scaled_values.shape", scaled_values.shape)
    values_to_optim_wel_prof = []
    for one_welfare_optimal_profile in welfare_optimal_profiles:
        one_welfare_optimal_profile = np.array(one_welfare_optimal_profile)
        one_welfare_optimal_profile = np.stack(
            [one_welfare_optimal_profile] * len(scaled_values), axis=0
        )

        print("one_welfare_optimal_profile", one_welfare_optimal_profile.shape)
        dist_ = np.linalg.norm(
            np.array(scaled_values) - np.array(one_welfare_optimal_profile),
            axis=1,
        )
        print("dist_", dist_.shape)
        values_to_optim_wel_prof.append(dist_)
    values_to_optim_wel_prof = np.stack(values_to_optim_wel_prof, axis=1)
    print("values_to_optim_wel_prof.shape", values_to_optim_wel_prof.shape)
    values_to_optim_wel_prof = values_to_optim_wel_prof.min(axis=1)
    print(
        "reduced values_to_optim_wel_prof.shape",
        values_to_optim_wel_prof.shape,
    )

    scaled_distance = values_to_optim_wel_prof / disagreement_to_optim_wel_prof
    return scaled_distance


def _compute_metric_4(
    payoffs_for_pi,
    payoffs_for_all_pi,
    file_data,
):
    # max_w { [ w(outcome) - w(disagreement) ]
    # / [ max_\pi w(\pi) - w(disagreement)] }

    metrics_for_all_welfares = []
    for welfare_fn in file_data.welfare_functions:
        welfares = welfare_fn(payoffs_for_pi)
        welfares_all_pi = welfare_fn(payoffs_for_all_pi)
        disagreement_profile = file_data.welfare_optim[1]
        disagreement_welfare = welfare_fn(np.array([disagreement_profile]))
        max_welfare = welfares_all_pi.max()
        metric_for_one_welfare = (welfares - disagreement_welfare) / (
            max_welfare - disagreement_welfare
        )

        metrics_for_all_welfares.append(metric_for_one_welfare)
    metrics_for_all_welfares = np.stack(metrics_for_all_welfares, axis=1)
    print("metrics_for_all_welfares.shape", metrics_for_all_welfares.shape)
    metrics = metrics_for_all_welfares.max(axis=1)
    return metrics


def _convert_str_of_list_to_list(str_of_list):
    return [
        float(v)
        for v in str_of_list.replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .split(",")
    ]


def _plot_bars(perf_per_mode_per_files, welfare_split):
    plt.figure(figsize=(10, 3))

    legend, x, groups = _plot_merged_players(
        perf_per_mode_per_files, plot_all=True, welfare_split=welfare_split
    )
    plt.xticks(x, groups, rotation=15)
    plt.ylabel(NORMALIZED_NAME)
    plt.ylim((0.0, 1.0))

    plt.legend(
        legend,
        frameon=True,
        bbox_to_anchor=(1.0, -0.23),
    )

    # Save the figure and show
    plt.tight_layout(rect=[0, -0.05, 1.0, 1.0])
    plt.savefig(f"bar_plot_vanilla_split_{welfare_split}.png")


def _plot_bars_separate(perf_per_mode_per_files, welfare_split):
    plt.figure(figsize=(10, 3))
    rotation = 15
    plt.subplot(121)
    _, x, groups = _plot_merged_players(
        perf_per_mode_per_files, mcp=False, welfare_split=welfare_split
    )
    plt.xticks(x, groups, rotation=rotation, ha="right")
    plt.ylabel(NORMALIZED_NAME)
    # plt.ylim((0.0, 1.0))

    plt.subplot(122)
    legend, x, groups = _plot_merged_players(
        perf_per_mode_per_files, mcp=True, welfare_split=welfare_split
    )
    plt.xticks(x, groups, rotation=rotation, ha="right")
    # plt.ylabel("Normalized scores")
    # plt.ylim((0.0, 1.0))

    if USE_DISTANCE_TO_WELFARE_OPTIM:
        if welfare_split:
            plt.tight_layout()
        else:
            plt.tight_layout()
        plt.legend(
            legend,
            frameon=True,
            bbox_to_anchor=(0.0, 0.00, -0.4, 1.0),
        )
    else:
        if welfare_split:
            if GLOBAL_CROSS_PLAY_IN_LOLA:
                plt.tight_layout(rect=[0, 0.20, 1.0, 1.0])
                plt.legend(
                    legend,
                    frameon=True,
                    bbox_to_anchor=(-0.4, -0.35),
                )
            else:
                plt.tight_layout(rect=[0, 0.14, 1.0, 1.0])
                plt.legend(
                    legend,
                    frameon=True,
                    bbox_to_anchor=(-0.4, -0.35),
                )
        else:
            plt.tight_layout(rect=[0, 0.07, 1.0, 1.0])
            plt.legend(
                legend,
                frameon=True,
                bbox_to_anchor=(-0.7, -0.30),
            )

    # Save the figure and show
    # if not USE_DISTANCE_TO_WELFARE_OPTIM:
    #     if welfare_split:
    #         plt.tight_layout(rect=[0, -0.07, 1.0, 1.0])
    #     else:
    #         plt.tight_layout(rect=[0, -0.05, 1.0, 1.0])
    plt.savefig(f"bar_plot_separated_split_{welfare_split}.png")


def _plot_merged_players(
    perf_per_mode_per_files,
    mcp: bool = None,
    plot_all=False,
    welfare_split=True,
):
    all_perf = [el.perf for el in perf_per_mode_per_files]
    groups = [f"{el.env} + {el.base_algo}" for el in perf_per_mode_per_files]
    groups = [group.strip(" + ") for group in groups]
    width = 0.1

    (
        self_play,
        cross_play,
        same_pref_perf,
        diff_pref_perf,
        self_play_err,
        cross_play_err,
        same_pref_perf_err,
        diff_pref_perf_err,
    ) = _preproces_values(all_perf)

    if plot_all:
        plt.text(1.08, 0.04, NA, fontdict={"fontsize": 10.0, "rotation": 90})
        plt.text(3.08, 0.04, NA, fontdict={"fontsize": 10.0, "rotation": 90})
    else:
        if not mcp:
            if welfare_split:
                plt.text(
                    1.08, 0.05, NA, fontdict={"fontsize": 10.0, "rotation": 90}
                )
                plt.text(
                    3.08, 0.05, NA, fontdict={"fontsize": 10.0, "rotation": 90}
                )
            self_play = self_play[:N_NO_MCP]
            cross_play = cross_play[:N_NO_MCP]
            same_pref_perf = same_pref_perf[:N_NO_MCP]
            diff_pref_perf = diff_pref_perf[:N_NO_MCP]
            self_play_err = self_play_err[:N_NO_MCP]
            cross_play_err = cross_play_err[:N_NO_MCP]
            same_pref_perf_err = same_pref_perf_err[:N_NO_MCP]
            diff_pref_perf_err = diff_pref_perf_err[:N_NO_MCP]
            groups = groups[:N_NO_MCP]
            # plt.text(
            #     1.35,
            #     -0.35,
            #     "a)",
            #     fontdict={"fontsize": 14.0, "weight": "bold"},
            # )
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
            # plt.text(
            #     1.85,
            #     -0.35,
            #     "b)",
            #     fontdict={"fontsize": 14.0, "weight": "bold"},
            # )

    x = np.arange(len(self_play))
    if welfare_split:
        # replace/remove None
        same_pref_perf_x = []
        diff_pref_perf_x = []
        new_diff_pref_perf = []
        new_diff_pref_perf_err = []
        new_same_pref_perf = []
        new_same_pref_perf_err = []
        rm_x = []
        rm_cross = []
        rm_cross_err = []
        for (i,(
            x_i,
            cross,
            cross_err,
            same_pref,
            same_pref_err,
            diff_pref,
            diff_pref_err,
        )) in enumerate(zip(
            x,
            cross_play,
            cross_play_err,
            same_pref_perf,
            same_pref_perf_err,
            diff_pref_perf,
            diff_pref_perf_err,)
        ):
            pass_ = False
            if GLOBAL_CROSS_PLAY_IN_LOLA:
                if mcp:
                    if i == 1 or i == 3:
                        pass_ =True
                else:
                    if i == 1 or i ==3:
                        pass_ = True
            if pass_:
                rm_x.append(x_i)
                rm_cross.append(cross)
                rm_cross_err.append(cross_err)
                continue
            same_pref_perf_x.append(x_i)
            if same_pref is None:
                assert same_pref_err is None
                new_same_pref_perf.append(cross)
                new_same_pref_perf_err.append(cross_err)
            else:
                new_same_pref_perf.append(same_pref)
                new_same_pref_perf_err.append(same_pref_err)
            if diff_pref is not None:
                assert diff_pref_err is not None
                diff_pref_perf_x.append(x_i)
                new_diff_pref_perf.append(diff_pref)
                new_diff_pref_perf_err.append(diff_pref_err)

            # same_pref_perf = [
            #     cross if same_pref is None else same_pref
            #     f
            # ]
            # same_pref_perf_err = [
            #     cross if same_pref is None else same_pref
            #     for cross, same_pref in zip(cross_play_err, same_pref_perf_err)
            # ]

        same_pref_perf = new_same_pref_perf
        same_pref_perf_err = new_same_pref_perf_err
        diff_pref_perf = new_diff_pref_perf
        diff_pref_perf_err = new_diff_pref_perf_err

    plt.bar(
        x - width * 1.0 - 0.02,
        self_play,
        width,
        yerr=self_play_err,
        color=COLORS[0],
        ecolor="black",
        capsize=3,
    )
    if welfare_split:
        if GLOBAL_CROSS_PLAY_IN_LOLA:
            plt.bar(
                np.array(rm_x) + width * 0.0,
                rm_cross,
                width,
                yerr=rm_cross_err,
                color=COLORS[1],
                ecolor="black",
                capsize=3,
            )

    if welfare_split:
        if welfare_split and GLOBAL_CROSS_PLAY_IN_LOLA:
            use_x = np.array(same_pref_perf_x)
        else:
            use_x = x
        plt.bar(
            use_x + width * 0.0,
            same_pref_perf,
            width,
            yerr=same_pref_perf_err,
            color=COLORS[2] if GLOBAL_CROSS_PLAY_IN_LOLA else COLORS[1],
            ecolor="black",
            capsize=3,
        )
        # if mcp and SPLIT_NEGO:
        #     # x = x[:-1]
        #     # diff_pref_perf = diff_pref_perf[:-1]
        #     # diff_pref_perf_err = diff_pref_perf_err[:-1]
        #     diff_pref_perf[-1] = 0
        #     diff_pref_perf_err[-1] = 0
        if welfare_split:
            use_x = np.array(diff_pref_perf_x)
        else:
            use_x = x
        plt.bar(
            use_x + width * 1.0 + 0.02,
            diff_pref_perf,
            width,
            yerr=diff_pref_perf_err,
            color=COLORS[3] if GLOBAL_CROSS_PLAY_IN_LOLA else COLORS[2],
            ecolor="black",
            capsize=3,
        )
        legend = LEGEND
    else:
        plt.bar(
            x + width * 0.0,
            cross_play,
            width,
            yerr=cross_play_err,
            color=COLORS[1],
            ecolor="black",
            capsize=3,
        )
        legend = LEGEND_NO_SPLIT


    if mcp and SPLIT_NEGO:
        # extra for negotiation env
        n_values = 6
        x_nego_shit = 4
        with_nego = width / 2  # - 0.001
        space_between_lines = 0.02
        negotiation_x = np.array(
            [i * with_nego + i * space_between_lines for i in range(n_values)]
        )
        diff_pref_perf_err = [diff_pref_perf_err[-1] for i in range(n_values)]

        # [0.50369, 0.49126, 0.46351, 0.44590, 0.42530, 0.41031]
        # [0.50626, 0.47991, 0.46098, 0.43771, 0.41802, 0.41276]

        negotiation_y = [0.47328, 0.44078, 0.41710, 0.39257, 0.37393, 0.31258]
        negotiation_y = [el / NEGOTIATION_RATIO for el in negotiation_y]
        diff_pref_perf_err = [
            0.00089,
            0.00088,
            0.00087,
            0.00088,
            0.00087,
            0.00087,
        ]
        negotiation_y = [
            el / NEGOTIATION_RATIO / np.sqrt(2) for el in negotiation_y
        ]

        plt.bar(
            x_nego_shit + negotiation_x + width * 1.0,
            negotiation_y,
            with_nego,
            yerr=diff_pref_perf_err,
            color=COLORS[2],
            ecolor="black",
            capsize=3,
        )

    if not welfare_split:
        # groups = [el.strip("*") if el.endswith(" *") else el for el in groups]
        groups = [
            el.strip("*")
            if el.endswith(" *")
            else el.replace("**", "*").replace("***", "**")
            for el in groups
        ]

    if REMOVE_STARS:
        groups = [el.replace("*", "").strip() for el in groups]

    return legend, x, groups


def _preproces_values(all_perf):
    self_play_p0 = _extract_value(all_perf, 0, PLAYER_0, "raw")
    self_play_p1 = _extract_value(all_perf, 0, PLAYER_1, "raw")
    cross_play_p0 = _extract_value(all_perf, 1, PLAYER_0, "raw")
    cross_play_p1 = _extract_value(all_perf, 1, PLAYER_1, "raw")
    same_pref_p0 = _extract_value(all_perf, 2, PLAYER_0, "raw")
    same_pref_p1 = _extract_value(all_perf, 2, PLAYER_1, "raw")
    diff_pref_p0 = _extract_value(all_perf, 3, PLAYER_0, "raw")
    diff_pref_p1 = _extract_value(all_perf, 3, PLAYER_1, "raw")

    if USE_DISTANCE_TO_WELFARE_OPTIM:
        self_play = self_play_p0
        cross_play = cross_play_p0
        same_pref_perf = same_pref_p0
        diff_pref_perf = diff_pref_p0
    else:
        self_play = _avg_over_players(self_play_p0, self_play_p1)
        cross_play = _avg_over_players(cross_play_p0, cross_play_p1)
        same_pref_perf = _avg_over_players(same_pref_p0, same_pref_p1)
        diff_pref_perf = _avg_over_players(diff_pref_p0, diff_pref_p1)

    _log_n_replicates(
        self_play,
        cross_play,
        same_pref_perf,
        diff_pref_perf,
    )

    self_play_err = _get_std_err(self_play)
    cross_play_err = _get_std_err(cross_play)
    same_pref_perf_err = _get_std_err(same_pref_perf)
    diff_pref_perf_err = _get_std_err(diff_pref_perf)

    self_play = _get_mean(self_play)
    cross_play = _get_mean(cross_play)
    same_pref_perf = _get_mean(same_pref_perf)
    diff_pref_perf = _get_mean(diff_pref_perf)

    self_play = _replace_final_values(all_perf, self_play, "self-play", "mean")
    cross_play = _replace_final_values(
        all_perf, cross_play, "cross-play", "mean"
    )
    same_pref_perf = _replace_final_values(
        all_perf, same_pref_perf, "cross-play same", "mean"
    )
    diff_pref_perf = _replace_final_values(
        all_perf, diff_pref_perf, "cross-play diff", "mean"
    )

    self_play_err = _replace_final_values(
        all_perf, self_play_err, "self-play", "std err"
    )
    cross_play_err = _replace_final_values(
        all_perf, cross_play_err, "cross-play", "std err"
    )
    same_pref_perf_err = _replace_final_values(
        all_perf, same_pref_perf_err, "cross-play same", "std err"
    )
    diff_pref_perf_err = _replace_final_values(
        all_perf, diff_pref_perf_err, "cross-play diff", "std err"
    )

    return (
        self_play,
        cross_play,
        same_pref_perf,
        diff_pref_perf,
        self_play_err,
        cross_play_err,
        same_pref_perf_err,
        diff_pref_perf_err,
    )


def _log_n_replicates(
    self_play,
    cross_play,
    same_pref_perf,
    diff_pref_perf,
):
    print("\n_log_n_replicates")
    print("self_play", [None if el is None else el.shape for el in self_play])
    print(
        "cross_play", [None if el is None else el.shape for el in cross_play]
    )
    print(
        "same_pref_perf",
        [None if el is None else el.shape for el in same_pref_perf],
    )
    print(
        "diff_pref_perf",
        [None if el is None else el.shape for el in diff_pref_perf],
    )

    ratio = []
    for cross, cross_same, cross_diff in zip(
        cross_play, same_pref_perf, diff_pref_perf
    ):
        # if len(cross_same.shape) > 0:
        if cross_same is not None:
            assert cross.shape[0] == (
                cross_same.shape[0] + cross_diff.shape[0]
            ), f"{cross.shape[0]} == {cross_same.shape[0]} + {cross_diff.shape[0]}"
            ratio.append(cross_same.shape[0] / cross_diff.shape[0])
        else:
            ratio.append(None)
    print("cross_same / cross_diff", ratio)


def _extract_value(all_perf, idx, player_idx, attrib):

    values = []
    for el in all_perf:
        if isinstance(el, Final_values):
            values.append(None)
        else:
            if hasattr(el[idx][player_idx], attrib):
                values.append(getattr(el[idx][player_idx], attrib))
            else:
                values.append(None)
    return values


def _avg_over_players(values_player0, values_player1):
    return [
        (np.array(v_p0) + np.array(v_p1)) / 2 if v_p0 is not None else None
        for v_p0, v_p1 in zip(values_player0, values_player1)
    ]


def _get_std_err(values):
    return [
        v.std() / np.sqrt(v.shape[0]) if v is not None else None
        for v in values
    ]


def _get_mean(values):
    return [v.mean() if v is not None else None for v in values]


def _replace_final_values(all_perf, values, cat, mode):
    new_values = []
    for el, value in zip(all_perf, values):
        if isinstance(el, Final_values):
            if mode == "mean":
                idx = 0
            elif mode == "std err":
                idx = 1
            else:
                raise ValueError()

            if cat == "self-play":
                v = el.self_play[idx]
            elif cat == "cross-play":
                v = el.cross_play[idx]
            elif cat == "cross-play same":
                v = el.cross_play_same[idx]
            elif cat == "cross-play diff":
                v = el.cross_play_diff[idx]
            else:
                raise ValueError()
        else:
            v = value
        new_values.append(v)
    return new_values


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
