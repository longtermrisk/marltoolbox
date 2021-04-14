import copy
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.utils import merge_dicts

from marltoolbox.algos import welfare_coordination
from marltoolbox.experiments.rllib_api import amtft_various_env
from marltoolbox.utils import (
    self_and_cross_perf,
    postprocessing,
    miscellaneous,
    plot,
)

logger = logging.getLogger(__name__)


def main(debug):
    if debug:
        test_meta_solver(debug)

    hp = _get_hyperparameters(debug)

    results = []
    for tau in hp["tau_range"]:
        (
            all_rllib_config,
            hp_eval,
            env_config,
            stop_config,
        ) = _produce_rllib_config_for_each_replicates(tau, hp)

        mixed_rllib_configs = _mix_rllib_config(all_rllib_config, hp_eval)

        tune_analysis = ray.tune.run(
            dqn.DQNTrainer,
            config=mixed_rllib_configs,
            verbose=1,
            stop=stop_config,
            checkpoint_at_end=True,
            name=hp_eval["exp_name"],
            log_to_file=not hp_eval["debug"],
            # loggers=None
            # if hp_eval["debug"]
            # else DEFAULT_LOGGERS + (WandbLogger,),
        )
        mean_player_1_payoffs, mean_player_2_payoffs = _extract_metrics(
            tune_analysis, hp_eval
        )

        results.append(
            (
                tau,
                (mean_player_1_payoffs, mean_player_2_payoffs),
            )
        )
    _save_to_json(exp_name=hp["exp_name"], object=results)
    _plot_results(exp_name=hp["exp_name"], results=results, hp_eval=hp_eval)


def _get_hyperparameters(debug):
    # env = "IteratedPrisonersDilemma"
    env = "IteratedAsymBoS"
    # env = "CoinGame"

    hp = amtft_various_env.get_hyperparameters(
        debug, train_n_replicates=1, filter_utilitarian=False, env=env
    )

    hp.update(
        {
            "n_replicates_over_full_exp": 2,
            "final_base_game_eval_over_n_epi": 2,
            "tau_range": np.arange(0.0, 1.1, 0.5)
            if hp["debug"]
            else np.arange(0.0, 1.1, 0.1),
            "n_self_play_in_final_meta_game": 0,
            "n_cross_play_in_final_meta_game": 1,
        }
    )
    return hp


def _produce_rllib_config_for_each_replicates(tau, hp):
    all_rllib_config = []
    for replicate_i in range(hp["n_replicates_over_full_exp"]):
        hp_eval = _load_base_game_results(
            copy.deepcopy(hp), load_base_replicate_i=replicate_i
        )
        hp_eval["tau"] = tau

        (
            rllib_config,
            hp_eval,
            env_config,
            stop_config,
        ) = _get_vanilla_amTFT_eval_config(
            hp_eval, hp_eval["final_base_game_eval_over_n_epi"]
        )

        rllib_config = _modify_config_to_use_welfare_coordinators(
            rllib_config, env_config, hp_eval
        )
        all_rllib_config.append(rllib_config)
    return all_rllib_config, hp_eval, env_config, stop_config


def _load_base_game_results(hp, load_base_replicate_i):
    prefix = "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-2/"
    # prefix = "/ray_results/"
    if "CoinGame" in hp["env_name"]:
        hp["data_dir"] = (prefix + "amTFT/2021_04_10/19_37_20",)[
            load_base_replicate_i
        ]
    elif "IteratedAsymBoS" in hp["env_name"]:
        hp["data_dir"] = (
            prefix + "amTFT/2021_04_13/11_56_23",  # 5 replicates
            prefix + "amTFT/2021_04_13/13_40_03",  # 5 replicates
            prefix + "amTFT/2021_04_13/13_40_34",  # 5 replicates
            prefix + "amTFT/2021_04_13/18_06_48",  # 10 replicates
            prefix + "amTFT/2021_04_13/18_07_05",  # 10 replicates
        )[load_base_replicate_i]
    elif "IteratedPrisonersDilemma" in hp["env_name"]:
        hp["data_dir"] = (
            "/home/maxime/dev-maxime/CLR/vm-data/"
            "instance-10-cpu-4/amTFT/2021_04_13/12_12_56",
        )[load_base_replicate_i]

    hp["json_file"] = _get_results_json_file_path_in_dir(hp["data_dir"])
    hp["ckpt_per_welfare"] = _get_checkpoints_for_each_welfare_in_dir(
        hp["data_dir"], hp
    )

    return hp


def _get_vanilla_amTFT_eval_config(hp, final_eval_over_n_epi):
    hp_eval = amtft_various_env.modify_hp_for_evaluation(
        hp, final_eval_over_n_epi
    )

    stop_config, env_config, rllib_config = amtft_various_env.get_rllib_config(
        hp_eval, welfare_fn=postprocessing.WELFARE_INEQUITY_AVERSION, eval=True
    )
    rllib_config = amtft_various_env.modify_config_for_evaluation(
        rllib_config, hp_eval, env_config
    )
    hp_eval["n_self_play_per_checkpoint"] = None
    hp_eval["n_cross_play_per_checkpoint"] = None
    hp_eval[
        "x_axis_metric"
    ] = f"policy_reward_mean.{env_config['players_ids'][0]}"
    hp_eval[
        "y_axis_metric"
    ] = f"policy_reward_mean.{env_config['players_ids'][1]}"

    return rllib_config, hp_eval, env_config, stop_config


def _modify_config_to_use_welfare_coordinators(
    rllib_config, env_config, hp_eval
):
    all_welfare_pairs_wt_payoffs = _get_all_welfare_pairs_wt_payoffs(
        hp_eval, env_config["players_ids"]
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
                # "own_default_welfare_fn": postprocessing.WELFARE_INEQUITY_AVERSION
                # if policy_idx
                # else postprocessing.WELFARE_UTILITARIAN,
                # "opp_default_welfare_fn": postprocessing.WELFARE_INEQUITY_AVERSION
                # if opp_policy_idx
                # else postprocessing.WELFARE_UTILITARIAN,
                "own_default_welfare_fn": "inequity aversion"
                if policy_idx
                else "utilitarian",
                "opp_default_welfare_fn": "inequity aversion"
                if opp_policy_idx
                else "utilitarian",
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


def _mix_rllib_config(all_rllib_configs, hp_eval):
    if (
        hp_eval["n_self_play_in_final_meta_game"] == 1
        and hp_eval["n_cross_play_in_final_meta_game"] == 0
    ):
        return all_rllib_configs[0]
    elif (
        hp_eval["n_self_play_in_final_meta_game"] == 0
        and hp_eval["n_cross_play_in_final_meta_game"] != 0
    ):
        master_config = _mix_configs_policies(
            all_rllib_configs,
            n_mix_per_config=hp_eval["n_cross_play_in_final_meta_game"],
        )
        return master_config
    else:
        raise ValueError()


def _mix_configs_policies(all_rllib_configs, n_mix_per_config):
    assert n_mix_per_config <= len(all_rllib_configs) - 1
    policy_ids = all_rllib_configs[0]["multiagent"]["policies"].keys()
    assert len(policy_ids) == 2
    _assert_all_config_use_the_same_policies(all_rllib_configs, policy_ids)

    policy_config_variants = _gather_policy_variant_per_policy_id(
        all_rllib_configs, policy_ids
    )

    master_config = _create_one_master_config(
        all_rllib_configs, policy_config_variants, policy_ids, n_mix_per_config
    )
    return master_config


def _create_one_master_config(
    all_rllib_configs, policy_config_variants, policy_ids, n_mix_per_config
):
    all_policy_mix = []
    player_1, player_2 = policy_ids
    for config_idx, p1_policy_config in enumerate(
        policy_config_variants[player_1]
    ):
        policies_mixes = _produce_n_mix_with_player_2_policies(
            policy_config_variants,
            player_2,
            config_idx,
            n_mix_per_config,
            player_1,
            p1_policy_config,
        )
        all_policy_mix.extend(policies_mixes)

    master_config = copy.deepcopy(all_rllib_configs[0])
    master_config["multiagent"]["policies"] = tune.grid_search(all_policy_mix)
    return master_config


def _produce_n_mix_with_player_2_policies(
    policy_config_variants,
    player_2,
    config_idx,
    n_mix_per_config,
    player_1,
    p1_policy_config,
):
    p2_policy_configs_sampled = _get_p2_policies_samples_excluding_self(
        policy_config_variants, player_2, config_idx, n_mix_per_config
    )
    policies_mixes = []
    for p2_policy_config in p2_policy_configs_sampled:
        policy_mix = {
            player_1: p1_policy_config,
            player_2: p2_policy_config,
        }
        policies_mixes.append(policy_mix)
    return policies_mixes


def _get_p2_policies_samples_excluding_self(
    policy_config_variants, player_2, config_idx, n_mix_per_config
):
    p2_policy_config_variants = copy.deepcopy(policy_config_variants[player_2])
    p2_policy_config_variants.pop(config_idx)
    p2_policy_configs_sampled = random.sample(
        p2_policy_config_variants, n_mix_per_config
    )
    return p2_policy_configs_sampled


def _assert_all_config_use_the_same_policies(all_rllib_configs, policy_ids):
    for rllib_config in all_rllib_configs:
        assert rllib_config["multiagent"]["policies"].keys() == policy_ids


def _gather_policy_variant_per_policy_id(all_rllib_configs, policy_ids):
    policy_config_variants = {}
    for policy_id in policy_ids:
        policy_config_variants[policy_id] = []
        for rllib_config in all_rllib_configs:
            policy_config_variants[policy_id].append(
                copy.deepcopy(
                    rllib_config["multiagent"]["policies"][policy_id]
                )
            )
    return policy_config_variants


def _extract_metrics(tune_analysis, hp_eval):
    player_1_payoffs = miscellaneous.extract_metric_values_per_trials(
        tune_analysis, metric=hp_eval["x_axis_metric"]
    )
    player_2_payoffs = miscellaneous.extract_metric_values_per_trials(
        tune_analysis, metric=hp_eval["y_axis_metric"]
    )
    mean_player_1_payoffs = sum(player_1_payoffs) / len(player_1_payoffs)
    mean_player_2_payoffs = sum(player_2_payoffs) / len(player_2_payoffs)
    print(
        "mean_player_1_payoffs, mean_player_2_payoffs",
        mean_player_1_payoffs,
        mean_player_2_payoffs,
    )
    return mean_player_1_payoffs, mean_player_2_payoffs


def _save_to_json(exp_name, object):
    exp_dir = os.path.join("~/ray_results", exp_name)
    exp_dir = os.path.expanduser(exp_dir)
    json_file = os.path.join(exp_dir, "final_eval_in_base_game.json")
    with open(json_file, "w") as outfile:
        json.dump(object, outfile)


def _plot_results(exp_name, results, hp_eval):
    exp_dir = os.path.join("~/ray_results", exp_name)
    exp_dir = os.path.expanduser(exp_dir)

    data_groups_per_mode = _format_result_for_plotting(results)

    if "CoinGame" in hp_eval["env_name"]:
        background_area_coord = None
    else:
        background_area_coord = hp_eval["env_class"].PAYOUT_MATRIX
    plot_config = plot.PlotConfig(
        save_dir_path=exp_dir,
        xlim=hp_eval["x_limits"],
        ylim=hp_eval["y_limits"],
        markersize=5,
        jitter=hp_eval["jitter"],
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        x_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][0],
        y_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][1],
        background_area_coord=background_area_coord,
    )

    plot_helper = plot.PlotHelper(plot_config)
    plot_helper.plot_dots(data_groups_per_mode)


def _format_result_for_plotting(results):
    data_groups_per_mode = {}
    for (
        tau,
        (mean_player_1_payoffs, mean_player_2_payoffs),
    ) in results:
        df_row_dict = {}
        df_row_dict[f"mean"] = (
            mean_player_1_payoffs,
            mean_player_2_payoffs,
        )
        df_rows = [df_row_dict]
        data_groups_per_mode[f"tau_{tau}"] = pd.DataFrame(df_rows)
    return data_groups_per_mode


def test_meta_solver(debug):
    hp = _get_hyperparameters(debug)
    hp = _load_base_game_results(hp, load_base_replicate_i=0)
    stop, env_config, rllib_config = amtft_various_env.get_rllib_config(
        hp, welfare_fn=postprocessing.WELFARE_INEQUITY_AVERSION, eval=True
    )
    rllib_config = _simulate_update_by_dqn_trainer(rllib_config)
    print('hp["env_class"]', hp["env_class"])

    for player_id in env_config["players_ids"]:
        print("====================")
        print("====================")
        print("For player_id:", player_id)
        meta_policy_config = _format_meta_policy_config(
            rllib_config, hp, player_id
        )
        print(
            'hp["env_class"](env_config).OBSERVATION_SPACE',
            hp["env_class"](env_config).OBSERVATION_SPACE,
        )
        print('hp["env_class"].ACTION_SPACE', hp["env_class"].ACTION_SPACE)
        player_meta_policy = (
            welfare_coordination.WelfareCoordinationTorchPolicy(
                hp["env_class"](env_config).OBSERVATION_SPACE,
                hp["env_class"].ACTION_SPACE,
                meta_policy_config,
            )
        )

        all_welfare_pairs_wt_payoffs = _get_all_welfare_pairs_wt_payoffs(
            hp, env_config["players_ids"]
        )
        print("all_welfare_pairs_wt_payoffs", all_welfare_pairs_wt_payoffs)
        player_meta_policy.setup_meta_game(
            copy.deepcopy(all_welfare_pairs_wt_payoffs),
            own_player_idx=0,
            opp_player_idx=1,
            own_default_welfare_fn="utilitarian",
            opp_default_welfare_fn="inequity aversion",
        )

        for tau_x_10 in range(0, 11, 1):
            player_meta_policy.solve_meta_game(tau_x_10 / 10)


def _simulate_update_by_dqn_trainer(rllib_config):
    explo = copy.deepcopy(rllib_config["exploration_config"])
    rllib_config = merge_dicts(dqn.DEFAULT_CONFIG, rllib_config)
    rllib_config["exploration_config"] = explo
    return rllib_config


def _format_meta_policy_config(rllib_config, hp, policy_id):
    policy_config = copy.deepcopy(
        rllib_config["multiagent"]["policies"][policy_id][3]
    )
    policy_class = copy.deepcopy(
        rllib_config["multiagent"]["policies"][policy_id][0]
    )
    meta_policy_config = copy.deepcopy(rllib_config)
    meta_policy_config.update(
        _convert_policy_config_to_meta_policy_config(
            hp, policy_id, policy_config, policy_class
        )
    )
    return meta_policy_config


def _get_results_json_file_path_in_dir(data_dir):
    eval_dir = _eval_dir(data_dir)
    json_result_path = _get_json_results_path(eval_dir)
    return json_result_path


def _eval_dir(data_dir):
    eval_dir_parents_2 = os.path.join(data_dir, "eval")
    eval_dir_parents_1 = _get_unique_child_dir(eval_dir_parents_2)
    eval_dir = _get_unique_child_dir(eval_dir_parents_1)
    return eval_dir


def _get_json_results_path(eval_dir):
    all_files_filtered = _get_children_paths_wt_selecting_filter(
        eval_dir, _filter=self_and_cross_perf.RESULTS_SUMMARY_FILENAME_PREFIX
    )
    all_files_filtered = [
        file_path for file_path in all_files_filtered if ".json" in file_path
    ]
    assert len(all_files_filtered) == 1, f"{all_files_filtered}"
    json_result_path = os.path.join(eval_dir, all_files_filtered[0])
    return json_result_path


def _get_unique_child_dir(_dir):
    list_dir = os.listdir(_dir)
    assert len(list_dir) == 1, f"{list_dir}"
    unique_child_dir = os.path.join(_dir, list_dir[0])
    return unique_child_dir


def _get_checkpoints_for_each_welfare_in_dir(data_dir, hp):
    ckpt_per_welfare = {}
    for welfare_fn, welfare_name in hp["welfare_functions"]:
        welfare_training_save_dir = os.path.join(data_dir, welfare_fn, "coop")
        all_replicates_save_dir = _get_dir_of_each_replicate(
            welfare_training_save_dir
        )
        ckpts = _get_checkpoint_for_each_replicates(all_replicates_save_dir)
        ckpt_per_welfare[welfare_name.replace("_", " ")] = ckpts
    return ckpt_per_welfare


def _get_dir_of_each_replicate(welfare_training_save_dir):
    return _get_children_paths_wt_selecting_filter(
        welfare_training_save_dir, _filter="DQN_"
    )


def _get_checkpoint_for_each_replicates(all_replicates_save_dir):
    ckpt_dir_per_replicate = []
    for replicate_dir_path in all_replicates_save_dir:
        ckpt_dir_path = _get_ckpt_dir_for_one_replicate(replicate_dir_path)
        ckpt_path = _get_ckpt_fro_ckpt_dir(ckpt_dir_path)
        ckpt_dir_per_replicate.append(ckpt_path)
    return ckpt_dir_per_replicate


def _get_ckpt_dir_for_one_replicate(replicate_dir_path):
    partialy_filtered_ckpt_dir = _get_children_paths_wt_selecting_filter(
        replicate_dir_path, _filter="checkpoint_"
    )
    ckpt_dir = [
        file_path
        for file_path in partialy_filtered_ckpt_dir
        if ".is_checkpoint" not in file_path
    ]
    assert len(ckpt_dir) == 1, f"{ckpt_dir}"
    return ckpt_dir[0]


def _get_ckpt_fro_ckpt_dir(ckpt_dir_path):
    partialy_filtered_ckpt_path = _get_children_paths_wt_discarding_filter(
        ckpt_dir_path, _filter="tune_metadata"
    )
    ckpt_path = [
        file_path
        for file_path in partialy_filtered_ckpt_path
        if ".is_checkpoint" not in file_path
    ]
    assert len(ckpt_path) == 1, f"{ckpt_path}"
    return ckpt_path[0]


def _get_children_paths_wt_selecting_filter(parent_dir_path, _filter):
    return _get_children_paths_filters(
        parent_dir_path, selecting_filter=_filter
    )


def _get_children_paths_wt_discarding_filter(parent_dir_path, _filter):
    return _get_children_paths_filters(
        parent_dir_path, discarding_filter=_filter
    )


def _get_children_paths_filters(
    parent_dir_path: str,
    selecting_filter: str = None,
    discarding_filter: str = None,
):
    filtered_children = os.listdir(parent_dir_path)
    if selecting_filter is not None:
        filtered_children = [
            filename
            for filename in filtered_children
            if selecting_filter in filename
        ]
    if discarding_filter is not None:
        filtered_children = [
            filename
            for filename in filtered_children
            if discarding_filter not in filename
        ]
    filtered_children_path = [
        os.path.join(parent_dir_path, filename)
        for filename in filtered_children
    ]
    return filtered_children_path


def _convert_policy_config_to_meta_policy_config(
    hp, policy_id, policy_config, policy_class
):
    meta_policy_config = welfare_coordination.DEFAULT_CONFIG
    meta_policy_config["solve_meta_game_after_init"] = False
    meta_policy_config["policy_id_to_load"] = policy_id
    meta_policy_config["nested_policies"] = [
        # You need to provide the policy class for every nested Policies
        {
            "Policy_class": policy_class,
            "config_update": policy_config,
        },
    ]
    meta_policy_config["policy_checkpoints"] = hp["ckpt_per_welfare"]

    return meta_policy_config


def _get_all_welfare_pairs_wt_payoffs(hp, player_ids):
    with open(hp["json_file"]) as json_file:
        json_data = json.load(json_file)

    cross_play_data = _keep_only_cross_play_values(json_data)
    cross_play_means = _keep_only_mean_values(cross_play_data)
    all_welfare_pairs_wt_payoffs = _order_players(cross_play_means, player_ids)

    return all_welfare_pairs_wt_payoffs


def _keep_only_cross_play_values(json_data):
    return {
        _format_eval_mode(eval_mode): v
        for eval_mode, v in json_data.items()
        if "cross-play" in eval_mode
    }


def _format_eval_mode(eval_mode):
    k_wtout_kind_of_play = eval_mode.split(":")[-1].strip()
    both_welfare_fn = k_wtout_kind_of_play.split(" vs ")
    return welfare_coordination.WelfareCoordinationTorchPolicy._from_pair_of_welfare_names_to_key(
        *both_welfare_fn
    )


def _keep_only_mean_values(cross_play_data):
    return {
        wekfare_pair_k: {
            player_k: player_v["mean"]
            for player_k, player_v in eval_dict.items()
        }
        for wekfare_pair_k, eval_dict in cross_play_data.items()
    }


def _order_players(cross_play_means, player_ids):
    return {
        wekfare_pair_k: [
            eval_dict[[k for k in eval_dict.keys() if player_ids[0] in k][0]],
            eval_dict[[k for k in eval_dict.keys() if player_ids[1] in k][0]],
        ]
        for wekfare_pair_k, eval_dict in cross_play_means.items()
    }


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
