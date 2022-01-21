import os, pickle
import numpy as np
import matplotlib.pyplot as plt


def _print_metrix_perf(
    self_play_payoff_matrices_7x7, cross_play_payoff_matrices_7x7
):
    for ia_coeff in range(7):
        print("ia_coeff player 0:", ia_coeff / 10)
        pl0_self_play_mean = self_play_payoff_matrices_7x7[
            ia_coeff, :, 0, :
        ].mean(axis=-1)
        pl1_self_play_mean = self_play_payoff_matrices_7x7[
            ia_coeff, :, 1, :
        ].mean(axis=-1)
        # print("Player 0 self_play_mean", pl0_self_play_mean)
        # print("Player 1 self_play_mean", pl1_self_play_mean)

        pl0_cross_play_mean = cross_play_payoff_matrices_7x7[
            ia_coeff, :, 0, :
        ].mean(axis=-1)
        pl1_cross_play_mean = cross_play_payoff_matrices_7x7[
            ia_coeff, :, 1, :
        ].mean(axis=-1)
        print("Player 0 cross_play_mean", pl0_cross_play_mean)
        print("Player 1 cross_play_mean", pl1_cross_play_mean)


def _print_perf_accross_differences(cross_play_payoff_matrices_7x7):
    cross_play_payoff_matrices_7x7 = cross_play_payoff_matrices_7x7.mean(
        axis=-1
    )
    normalization_factor = 0.5
    cross_play_same_pref = (
        (
            np.diag(cross_play_payoff_matrices_7x7[..., 0]).mean()
            + np.diag(cross_play_payoff_matrices_7x7[..., 1]).mean()
        )
        / 2
        / normalization_factor
    )
    print("cross_play_same_pref", cross_play_same_pref)
    mat_len = cross_play_payoff_matrices_7x7.shape[0]
    dia = np.diag_indices(mat_len)
    dia_sum = np.sum(cross_play_payoff_matrices_7x7[dia])
    cross_play_same_pref_bis = (
        np.mean(cross_play_payoff_matrices_7x7[dia]) / normalization_factor
    )
    off_dia_sum = np.sum(cross_play_payoff_matrices_7x7) - dia_sum
    cross_play_diff_pref = (
        off_dia_sum / (mat_len * (mat_len - 1)) / 2 / normalization_factor
    )

    print("cross_play_same_pref_bis", cross_play_same_pref_bis)
    print("cross_play_diff_pref", cross_play_diff_pref)

    # print("cross_play_payoff_matrices_7x7", cross_play_payoff_matrices_7x7)
    # plt.plot(cross_play_payoff_matrices_7x7[..., 0])
    # plt.show()

    cross_play_wt_diff_pref_diff_range = []
    for i in range(mat_len - 1):
        print("i", i + 1)
        cross_play_diff_pref = (
            np.mean(cross_play_payoff_matrices_7x7[0, i + 1, :])
            / normalization_factor
        )
        print("diff i")
        print("cross_play_diff_pref", cross_play_diff_pref)
        cross_play_wt_diff_pref_diff_range.append(cross_play_diff_pref)
    plt.plot(cross_play_wt_diff_pref_diff_range)
    plt.xlabel("IA coeff of the 2nd player")
    plt.ylabel("Normalized scores with IA coeff 1st player = 0.0")
    plt.show()

    dia_pos = list(dia)
    dia_neg = list(dia)
    cross_play_wt_diff_pref_diff_range = []
    for i in range(mat_len):

        pos_idx_kept = np.logical_and(
            dia_pos[1] >= 0, dia_pos[1] <= mat_len - 1
        )
        filtered_dia_pos = dia_pos[0][pos_idx_kept], dia_pos[1][pos_idx_kept]
        neg_idx_kept = np.logical_and(
            dia_neg[1] >= 0, dia_neg[1] <= mat_len - 1
        )
        filtered_dia_neg = dia_neg[0][neg_idx_kept], dia_neg[1][neg_idx_kept]
        print("i", i, "filtered_dia_pos", filtered_dia_pos)
        print("i", i, "filtered_dia_neg", filtered_dia_neg)
        cross_play_diff_pref = (
            (
                np.mean(cross_play_payoff_matrices_7x7[filtered_dia_pos])
                + np.mean(cross_play_payoff_matrices_7x7[filtered_dia_neg])
            )
            / 2
            / normalization_factor
        )
        print("diff i")
        print("cross_play_diff_pref", cross_play_diff_pref)
        cross_play_wt_diff_pref_diff_range.append(cross_play_diff_pref)

        dia_pos[1] = dia_pos[1] + 1
        dia_neg[1] = dia_neg[1] - 1
    plt.plot(cross_play_wt_diff_pref_diff_range)
    plt.xlabel("abs(IA coeff 1st player - IA coeff 2nd player)")
    plt.ylabel("Normalized scores")
    plt.show()


if __name__ == "__main__":

    data_prefix = (
        "/home/maxime/ssd_space/CLR/marltoolbox/marltoolbox/experiments"
        "/tune_class_api/"
    )

    file_path = os.path.join(
        data_prefix, "empirical_game_matrices_prosociality_coeff_0.3"
    )
    with open(file_path, "rb") as f:
        payoff_matrix_35x35 = pickle.load(f)

    self_play_payoff_matrices_7x7 = []
    cross_play_payoff_matrices_7x7 = []
    size = 7
    for i in range(5):
        for j in range(5):
            sub_mat = payoff_matrix_35x35[
                i * size : (i + 1) * size, j * size : (j + 1) * size, :
            ]
            if i == j:
                self_play_payoff_matrices_7x7.append(sub_mat)
            else:
                cross_play_payoff_matrices_7x7.append(sub_mat)

    self_play_payoff_matrices_7x7 = np.stack(
        self_play_payoff_matrices_7x7, axis=-1
    )
    cross_play_payoff_matrices_7x7 = np.stack(
        cross_play_payoff_matrices_7x7, axis=-1
    )

    _print_metrix_perf(
        self_play_payoff_matrices_7x7, cross_play_payoff_matrices_7x7
    )
    _print_perf_accross_differences(cross_play_payoff_matrices_7x7)
