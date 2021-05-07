##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the
# tune_class_api/lola_pg_official.py file
##########

import copy
import logging
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer

from marltoolbox import utils
from marltoolbox.algos import welfare_coordination
from marltoolbox.experiments.rllib_api import amtft_meta_game
from marltoolbox.experiments.tune_class_api import lola_exact_official
from marltoolbox.utils import (
    cross_play,
    restore,
    path,
    callbacks,
    log,
    miscellaneous,
)

logger = logging.getLogger(__name__)


EGALITARIAN = "egalitarian"
MIXED = "mixed"
UTILITARIAN = "utilitarian"
FAILURE = "failure"


def main(debug):
    # amtft_meta_game._extract_stats_on_welfare_announced(
    #     players_ids=["player_row", "player_col"],
    #     exp_dir="/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-2"
    #     "/LOLA_Exact/2021_04_21/17_53_53",
    # nested_info=True,
    # )

    hp = get_hyperparameters(debug)

    results = []
    ray.init(num_cpus=os.cpu_count(), local_mode=hp["debug"])
    for tau in hp["tau_range"]:
        hp["tau"] = tau
        (
            all_rllib_config,
            hp_eval,
            env_config,
            stop_config,
        ) = _produce_rllib_config_for_each_replicates(hp)

        mixed_rllib_configs = (
            cross_play.utils.mix_policies_in_given_rllib_configs(
                all_rllib_config, hp_eval["n_cross_play_in_final_meta_game"]
            )
        )

        tune_analysis = ray.tune.run(
            PGTrainer,
            config=mixed_rllib_configs,
            verbose=1,
            stop=stop_config,
            name=hp_eval["exp_name"],
            log_to_file=not hp_eval["debug"],
        )

        (
            mean_player_1_payoffs,
            mean_player_2_payoffs,
            player_1_payoffs,
            player_2_payoffs,
        ) = amtft_meta_game.extract_metrics(tune_analysis, hp_eval)

        results.append(
            (
                tau,
                (mean_player_1_payoffs, mean_player_2_payoffs),
                (player_1_payoffs, player_2_payoffs),
            )
        )
    amtft_meta_game.save_to_json(exp_name=hp["exp_name"], object=results)
    amtft_meta_game.plot_results(
        exp_name=hp["exp_name"],
        results=results,
        hp_eval=hp_eval,
        format_fn=amtft_meta_game.format_result_for_plotting,
    )
    amtft_meta_game.extract_stats_on_welfare_announced(
        players_ids=env_config["players_ids"],
        exp_name=hp["exp_name"],
        nested_info=True,
    )


def get_hyperparameters(debug):
    """Get hyperparameters for meta game with LOLA-Exact policies in base
    game"""
    # env = "IPD"
    env = "IteratedAsymBoS"

    hp = lola_exact_official.get_hyperparameters(
        debug, train_n_replicates=1, env=env
    )

    hp.update(
        {
            "n_replicates_over_full_exp": 2 if debug else 20,
            "final_base_game_eval_over_n_epi": 1 if debug else 200,
            "tau_range": np.arange(0.0, 1.1, 0.5)
            if hp["debug"]
            else np.arange(0.0, 1.1, 0.1),
            "n_self_play_in_final_meta_game": 0,
            "n_cross_play_in_final_meta_game": 1 if debug else 10,
            "welfare_functions": [
                (EGALITARIAN, EGALITARIAN),
                (MIXED, MIXED),
                (UTILITARIAN, UTILITARIAN),
            ],
        }
    )
    return hp


def _produce_rllib_config_for_each_replicates(hp):
    all_rllib_config = []
    for replicate_i in range(hp["n_replicates_over_full_exp"]):
        hp_eval = _load_base_game_results(
            copy.deepcopy(hp), load_base_replicate_i=replicate_i
        )

        (
            rllib_config,
            hp_eval,
            env_config,
            stop_config,
        ) = _get_vanilla_lola_exact_eval_config(
            hp_eval, hp_eval["final_base_game_eval_over_n_epi"]
        )

        rllib_config = _modify_config_to_use_welfare_coordinators(
            rllib_config, env_config, hp_eval
        )
        all_rllib_config.append(rllib_config)
    return all_rllib_config, hp_eval, env_config, stop_config


def _load_base_game_results(hp, load_base_replicate_i):

    # In local machine
    # prefix = "~/dev-maxime/CLR/vm-data/instance-10-cpu-2/"
    # prefix = "~/dev-maxime/CLR/vm-data/instance-10-cpu-2/"
    # prefix = "~/dev-maxime/CLR/vm-data/instance-60-cpu-2-preemtible/"
    prefix = "~/dev-maxime/CLR/vm-data/instance-60-cpu-3-preemtible/"
    prefix2 = "~/dev-maxime/CLR/vm-data/instance-60-cpu-4-preemtible/"

    # In VM
    # prefix = "~/ray_results/"
    # prefix2 = prefix

    prefix = os.path.expanduser(prefix)
    prefix2 = os.path.expanduser(prefix2)
    if "IteratedAsymBoS" in hp["env_name"]:
        hp["data_dir"] = (
            # instance-60-cpu-3-preemtible & instance-60-cpu-4-preemtible
            prefix + "LOLA_Exact/2021_05_05/14_49_18",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/14_50_39",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/14_51_01",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/14_53_56",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/14_56_32",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/15_46_08",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/15_46_23",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/15_46_59",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/15_47_22",  # 30 replicates
            prefix + "LOLA_Exact/2021_05_05/15_48_22",  # 30 replicates
            # instance-60-cpu-4-preemtible
            prefix2 + "LOLA_Exact/2021_05_07/07_52_32",  # 30 replicates
            # prefix2 + "LOLA_Exact/2021_05_07/08_02_21",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/08_02_38",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/08_02_49",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/08_03_03",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/08_54_58",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/08_55_34",  # 30 replicates
            # prefix2 + "LOLA_Exact/2021_05_07/08_56_00",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/09_04_07",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/09_09_30",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/09_09_42",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/10_02_15",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/10_02_30",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/10_02_39",  # 30 replicates
            prefix2 + "LOLA_Exact/2021_05_07/10_02_50",  # 30 replicates
        )[load_base_replicate_i]
    else:
        raise ValueError(f'bad env_name: {hp["env_name"]}')

    assert os.path.exists(hp["data_dir"]), (
        "Path doesn't exist. Probably that the prefix need to "
        f"be changed to fit the current machine used. path: {hp['data_dir']}"
    )

    print("==== Going to process data_dir", hp["data_dir"], "====")

    hp["ckpt_per_welfare"] = _get_checkpoints_for_each_welfare_in_dir(
        hp["data_dir"], hp
    )

    return hp


def _get_checkpoints_for_each_welfare_in_dir(data_dir, hp):
    all_replicates_save_dir = amtft_meta_game.get_dir_of_each_replicate(
        data_dir, str_in_dir="LOLAExactTrainer_"
    )
    assert len(all_replicates_save_dir) > 0
    welfares = _classify_base_replicates_into_welfares(all_replicates_save_dir)

    ckpt_per_welfare = {}
    for welfare_fn, welfare_name in hp["welfare_functions"]:
        replicates_save_dir_for_welfare = _filter_replicate_dir_by_welfare(
            all_replicates_save_dir, welfares, welfare_name
        )
        ckpts = restore.get_checkpoint_for_each_replicates(
            replicates_save_dir_for_welfare
        )
        ckpt_per_welfare[welfare_name] = [ckpt + ".json" for ckpt in ckpts]
    return ckpt_per_welfare


def _classify_base_replicates_into_welfares(all_replicates_save_dir):
    welfares = []
    for replicate_dir in all_replicates_save_dir:
        reward_player_1, reward_player_2 = _get_last_episode_rewards(
            replicate_dir
        )
        welfare_name = classify_into_welfare_based_on_rewards(
            reward_player_1, reward_player_2
        )
        welfares.append(welfare_name)
    return welfares


def classify_into_welfare_based_on_rewards(reward_player_1, reward_player_2):

    ratio = reward_player_1 / reward_player_2
    if ratio < 1.5:
        return EGALITARIAN
    elif ratio < 2.5:
        return MIXED
    else:
        return UTILITARIAN


def _filter_replicate_dir_by_welfare(
    all_replicates_save_dir, welfares, welfare_name
):
    replicates_save_dir_for_welfare = [
        replicate_dir
        for welfare, replicate_dir in zip(welfares, all_replicates_save_dir)
        if welfare == welfare_name
    ]
    return replicates_save_dir_for_welfare


def _get_last_episode_rewards(replicate_dir):
    results = utils.path.get_results_for_replicate(replicate_dir)
    last_epsiode_results = results[-1]
    return last_epsiode_results["ret1"], last_epsiode_results["ret2"]


def _get_vanilla_lola_exact_eval_config(hp, final_eval_over_n_epi):
    (
        hp_eval,
        rllib_config,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    ) = lola_exact_official.generate_eval_config(hp)

    hp_eval["n_self_play_per_checkpoint"] = None
    hp_eval["n_cross_play_per_checkpoint"] = None
    hp_eval[
        "x_axis_metric"
    ] = f"policy_reward_mean/{env_config['players_ids'][0]}"
    hp_eval[
        "y_axis_metric"
    ] = f"policy_reward_mean/{env_config['players_ids'][1]}"
    hp_eval["plot_axis_scale_multipliers"] = (
        1 / hp_eval["trace_length"],
        1 / hp_eval["trace_length"],
    )
    hp_eval["num_episodes"] = final_eval_over_n_epi
    stop_config["episodes_total"] = final_eval_over_n_epi
    rllib_config["callbacks"] = callbacks.merge_callbacks(
        callbacks.PolicyCallbacks,
        log.get_logging_callbacks_class(
            log_full_epi=True,
            log_full_epi_interval=1,
            log_from_policy_in_evaluation=True,
        ),
    )
    rllib_config["seed"] = miscellaneous.get_random_seeds(1)[0]
    rllib_config["log_level"] = "INFO"

    return rllib_config, hp_eval, env_config, stop_config


def _modify_config_to_use_welfare_coordinators(
    rllib_config, env_config, hp_eval
):
    all_welfare_pairs_wt_payoffs = (
        _get_all_welfare_pairs_wt_cross_play_payoffs(
            hp_eval, env_config["players_ids"]
        )
    )

    rllib_config["multiagent"]["policies_to_train"] = ["None"]
    policies = rllib_config["multiagent"]["policies"]
    for policy_idx, policy_id in enumerate(env_config["players_ids"]):
        policy_config_items = list(policies[policy_id])
        opp_policy_idx = (policy_idx + 1) % 2

        meta_policy_config = copy.deepcopy(welfare_coordination.DEFAULT_CONFIG)
        meta_policy_config.update(
            {
                "nested_policies": [
                    {
                        "Policy_class": copy.deepcopy(policy_config_items[0]),
                        "config_update": copy.deepcopy(policy_config_items[3]),
                    },
                ],
                "solve_meta_game_after_init": True,
                "tau": hp_eval["tau"],
                "all_welfare_pairs_wt_payoffs": all_welfare_pairs_wt_payoffs,
                "own_player_idx": policy_idx,
                "opp_player_idx": opp_policy_idx,
                "own_default_welfare_fn": EGALITARIAN
                if policy_idx == 1
                else UTILITARIAN,
                "opp_default_welfare_fn": EGALITARIAN
                if opp_policy_idx == 1
                else UTILITARIAN,
                "policy_id_to_load": policy_id,
                "policy_checkpoints": hp_eval["ckpt_per_welfare"],
            }
        )
        policy_config_items[
            0
        ] = welfare_coordination.WelfareCoordinationTorchPolicy
        policy_config_items[3] = meta_policy_config
        policies[policy_id] = tuple(policy_config_items)

    return rllib_config


def _get_all_welfare_pairs_wt_cross_play_payoffs(hp, player_ids):
    all_eval_replicates_dirs = _get_list_of_replicates_path_in_eval(hp)

    raw_data_points_wt_welfares = {}
    for eval_replicate_path in all_eval_replicates_dirs:
        players_ckpts = _extract_checkpoints_used_for_each_players(
            player_ids, eval_replicate_path
        )
        if _is_cross_play(players_ckpts):
            players_welfares = _convert_checkpoint_names_to_welfares(
                hp, players_ckpts
            )
            raw_players_perf = _extract_performance(
                eval_replicate_path, player_ids
            )
            play_mode = _get_play_mode(players_welfares)
            if play_mode not in raw_data_points_wt_welfares.keys():
                raw_data_points_wt_welfares[play_mode] = []
            raw_data_points_wt_welfares[play_mode].append(raw_players_perf)
    all_welfare_pairs_wt_payoffs = _average_perf_per_play_mode(
        raw_data_points_wt_welfares, hp
    )
    print("all_welfare_pairs_wt_payoffs", all_welfare_pairs_wt_payoffs)
    return all_welfare_pairs_wt_payoffs


def _get_list_of_replicates_path_in_eval(hp):
    child_dirs = utils.path.get_children_paths_wt_discarding_filter(
        hp["data_dir"], _filter="LOLAExact"
    )
    child_dirs = utils.path.keep_dirs_only(child_dirs)
    assert len(child_dirs) == 1, f"{child_dirs}"
    eval_dir = utils.path.get_unique_child_dir(child_dirs[0])
    eval_replicates_dir = utils.path.get_unique_child_dir(eval_dir)
    possible_nested_dir = utils.path.try_get_unique_child_dir(
        eval_replicates_dir
    )
    if possible_nested_dir is not None:
        eval_replicates_dir = possible_nested_dir
    all_eval_replicates_dirs = (
        utils.path.get_children_paths_wt_selecting_filter(
            eval_replicates_dir, _filter="PG_"
        )
    )
    return all_eval_replicates_dirs


def _extract_checkpoints_used_for_each_players(
    player_ids, eval_replicate_path
):
    params = utils.path.get_params_for_replicate(eval_replicate_path)
    policies_config = params["multiagent"]["policies"]
    ckps = [
        policies_config[player_id][3]["checkpoint_to_load_from"][0]
        for player_id in player_ids
    ]
    return ckps


def _is_cross_play(players_ckpts):
    return players_ckpts[0] != players_ckpts[1]


def _convert_checkpoint_names_to_welfares(hp, players_ckpts):
    players_welfares = []
    for player_ckpt in players_ckpts:
        player_ckpt_wtout_root = "/".join(player_ckpt.split("/")[-4:])
        for welfare, ckpts_for_welfare in hp["ckpt_per_welfare"].items():
            if any(
                player_ckpt_wtout_root in ckpt for ckpt in ckpts_for_welfare
            ):
                players_welfares.append(welfare)
                break

    assert len(players_welfares) == len(
        players_ckpts
    ), f"{len(players_welfares)} == {len(players_ckpts)}"
    return players_welfares


def _extract_performance(eval_replicate_path, player_ids):
    results_per_epi = utils.path.get_results_for_replicate(eval_replicate_path)
    players_avg_reward = _extract_and_average_perf(results_per_epi, player_ids)
    return players_avg_reward


def _extract_and_average_perf(results_per_epi, player_ids):
    players_avg_reward = []
    for player_id in player_ids:
        player_rewards = []
        for result_in_one_epi in results_per_epi:
            total_player_reward_in_one_epi = result_in_one_epi[
                "policy_reward_mean"
            ][player_id]
            player_rewards.append(total_player_reward_in_one_epi)
        players_avg_reward.append(sum(player_rewards) / len(player_rewards))
    return players_avg_reward


def _get_play_mode(players_welfares):
    return f"{players_welfares[0]}-{players_welfares[1]}"


def _average_perf_per_play_mode(raw_data_points_wt_welfares, hp):
    all_welfare_pairs_wt_payoffs = {}
    for (
        play_mode,
        values_per_replicates,
    ) in raw_data_points_wt_welfares.items():
        player_1_values = [
            value_replicate[0] for value_replicate in values_per_replicates
        ]
        player_2_values = [
            value_replicate[1] for value_replicate in values_per_replicates
        ]
        all_welfare_pairs_wt_payoffs[play_mode] = (
            sum(player_1_values) / len(player_1_values) / hp["trace_length"],
            sum(player_2_values) / len(player_2_values) / hp["trace_length"],
        )
    return all_welfare_pairs_wt_payoffs


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
