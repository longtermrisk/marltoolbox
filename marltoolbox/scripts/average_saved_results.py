import pickle
import os
import numpy as np
import json


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    for file, file_data in zip(files_to_process, files_data):
        (
            mean_per_player,
            std_dev_per_player,
            std_err_per_player,
        ) = _get_stats_for_file(file, n_players, file_data)

        print(
            file_data[0],
            "mean:",
            mean_per_player,
            "std_dev:",
            std_dev_per_player,
            "std_err:",
            std_err_per_player,
        )


def _get_inputs():
    prefix = "~/dev-maxime/CLR/vm-data/"
    files_data = (
        (  # 20 x 10 replicates in meta wt 30 x 5 replciates in base
            "META(PG)                           & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_07/13_46_27/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replciates in base
            "META(LOLA-Exact)                   & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_07/12_32_57/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replciates in base
            "META(alpha-rank pure strategies)   & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_07/12_05_00/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replciates in base
            "META(alpha-rank mixed strategies)  & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_07/12_00_06/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replciates in base
            "META(replicator dynamic)           & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/2021_05_07/11_24_50/final_base_game/final_eval_in_base_game.json",
        ),
    )
    n_players = 2
    return prefix, files_data, n_players


def _preprocess_inputs(prefix, files_data):
    files_to_process = [
        os.path.join(prefix, file_data[2]) for file_data in files_data
    ]
    return files_to_process


def _get_stats_for_file(file, n_players, file_data):
    file_path = os.path.expanduser(file)
    with (open(file_path, "rb")) as f:
        # file_content = pickle.load(f)
        file_content = json.load(f)
        values_per_replicat_per_player = np.array(file_content)

        assert values_per_replicat_per_player.ndim == 2
        n_replicates_in_content = values_per_replicat_per_player.shape[0]
        n_players_in_content = values_per_replicat_per_player.shape[1]
        assert n_players_in_content == n_players

        values_per_replicat_per_player = (
            values_per_replicat_per_player / file_data[1]
        )

        mean_per_player = values_per_replicat_per_player.mean(axis=0)
        std_dev_per_player = values_per_replicat_per_player.std(axis=0)
        std_err_per_player = std_dev_per_player / np.sqrt(
            n_replicates_in_content
        )
    return mean_per_player, std_dev_per_player, std_err_per_player


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
