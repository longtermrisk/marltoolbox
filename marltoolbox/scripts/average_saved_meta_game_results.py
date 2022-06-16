import json
import os

import numpy as np


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    for file, file_data in zip(files_to_process, files_data):
        (
            mean_per_player,
            std_dev_per_player,
            std_err_per_player,
            coordination_success,
        ) = _get_stats_for_file(file, n_players, file_data)

        print(
            file_data[0],
            "mean:",
            mean_per_player,
            "std_dev:",
            std_dev_per_player,
            "std_err:",
            std_err_per_player,
            "mean coordination_success:",
            coordination_success,
        )


def _get_inputs():
    # Files on Maxime's local machine
    # prefix = "~/dev-maxime/CLR/vm-data/"
    # Files in unzipped folder
    prefix = "./data/"
    files_data = (
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(PG)                           & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_07/13_46_27/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(LOLA-Exact)                   & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_07/12_32_57/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(alpha-rank pure strategies)   & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_07/12_05_00/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(alpha-rank mixed strategies)  & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_07/12_00_06/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(replicator dynamic)           & BASE(LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_07/11_24_50/final_base_game/final_eval_in_base_game.json",
        ),
        #
        #
        # BELOW WT ANNOUNCEMENT
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(Uniform(announcement+tau=0))  & BASE(announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/LOLA_Exact/2021_05_14/13_50_55"
            "/final_eval_in_base_game.json",
            "2nd_format_placeholder",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(alpha-rank mixed on welfare sets)  & BASE(announcement + "
            "LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/10_37_24"
            "/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(alpha-rank pure on welfare sets)  & BASE(announcement + "
            "LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/10_39_47/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(replicator dynamic random init on welfare sets)  & BASE("
            "announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/10_42_10/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(replicator dynamic default init on welfare sets)  & BASE("
            "announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/10_46_23/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(baseline random)  & BASE(" "announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/10_50_36"
            "/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(PG)  & BASE(" "announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/10_52_43/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(LOLA-Exact)  & BASE(" "announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/11_00_02"
            "/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(SOS-Exact)  & BASE(" "announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_14/12_38_59"
            "/meta_game/final_base_game/final_eval_in_base_game.json",
        ),
        (  # 20 x 10 replicates in meta wt 30 x 5 replicates in base
            "META(Minimum)  & BASE(" "announcement + LOLA-Exact)",
            200,
            "instance-60-cpu-4-preemtible/meta_game_compare/"
            "2021_05_27"
            "/19_24_36/meta_game/final_base_game/final_eval_in_base_game.json",
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
        file_content = json.load(f)
        file_content = _format_2nd_into_1st_format(file_content, file_data)
        if isinstance(file_content, dict):
            coordination_success = file_content["mean_coordination_success"]
            file_content = file_content["results"]
        else:
            coordination_success = (
                "N.A. => need to use the more recent "
                "saves stored under date 2021_05_26"
            )
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
    return (
        mean_per_player,
        std_dev_per_player,
        std_err_per_player,
        coordination_success,
    )


def _format_2nd_into_1st_format(file_content, file_data):
    if len(file_data) == 4:
        file_content = file_content[0][2]
        new_format = []
        for p1_content, p2_content in zip(file_content[0], file_content[1]):
            new_format.append((p1_content, p2_content))
        file_content = new_format
    return file_content


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)