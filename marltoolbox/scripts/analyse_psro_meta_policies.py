import json
import os

import numpy as np
from marltoolbox.utils import path


def main():
    prefix, training_folders, n_players = _get_inputs()
    replicates_folders_per_exp = _preprocess_inputs(prefix, training_folders)

    for replicates_folders, (_, n_base_policies) in zip(
            replicates_folders_per_exp,
                                  training_folders):
        _print_stats_for_exp(replicates_folders, n_base_policies)


def _get_inputs():
    prefix = "~/dev-maxime/CLR/vm-data/"
    training_folders = [
        ("instance-60-cpu-2-preemtible/PSRO_hardcoded/2021_05_26/11_49_58/",
         10)
    ]
    n_players = 2
    return prefix, training_folders, n_players


def _preprocess_inputs(prefix, training_folders):
    replicates_folders_per_exp = []
    for training_folder, _ in training_folders:
        training_folder_path = os.path.join(prefix, training_folder)
        training_folder_path = os.path.expanduser(training_folder_path)
        all_replicates = path.get_children_paths_wt_selecting_filter(
            training_folder_path, _filter="PSROTrainer"
        )
        print("all_replicates", len(all_replicates))
        replicates_folders_per_exp.append(all_replicates)

    return replicates_folders_per_exp


def _print_stats_for_exp(replicates_folders, n_base_policies):

    for replicates_folder in replicates_folders:
        results = path.get_results_for_replicate(replicates_folder)
        last_psro_iter_results = results[-1]
        print("last_psro_ier_results", last_psro_iter_results)
        player_0_meta_policy = last_psro_iter_results["player_0_meta_policy"]
        player_1_meta_policy = last_psro_iter_results["player_1_meta_policy"]
        for i in range(n_base_policies):

    # file_path = os.path.expanduser(file)
    # with (open(file_path, "rb")) as f:
    #     file_content = json.load(f)
    #     file_content = _format_2nd_into_1st_format(file_content, file_data)
    #     values_per_replicat_per_player = np.array(file_content)
    #
    #     assert values_per_replicat_per_player.ndim == 2
    #     n_replicates_in_content = values_per_replicat_per_player.shape[0]
    #     n_players_in_content = values_per_replicat_per_player.shape[1]
    #     assert n_players_in_content == n_players
    #
    #     values_per_replicat_per_player = (
    #         values_per_replicat_per_player / file_data[1]
    #     )
    #
    #     mean_per_player = values_per_replicat_per_player.mean(axis=0)
    #     std_dev_per_player = values_per_replicat_per_player.std(axis=0)
    #     std_err_per_player = std_dev_per_player / np.sqrt(
    #         n_replicates_in_content
    #     )
    # return mean_per_player, std_dev_per_player, std_err_per_player


if __name__ == "__main__":
    main()
