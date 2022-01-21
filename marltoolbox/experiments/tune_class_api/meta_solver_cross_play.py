import copy
import json
import os
import random
import torch
import ray
from ray import tune

from marltoolbox.experiments.tune_class_api import (
    lola_exact_official,
    various_algo_meta_game,
)
from marltoolbox.utils import log, miscellaneous, exp_analysis
from marltoolbox.experiments.tune_class_api.various_algo_meta_game import (
    META_UNIFORM,
    META_SOS,
    META_RANDOM,
    META_LOLA_EXACT,
    META_PG,
    META_APLHA_RANK,
    META_APLHA_PURE,
    META_REPLICATOR_DYNAMIC,
    META_REPLICATOR_DYNAMIC_ZERO_INIT,
    BASE_NEGOTIATION,
    BASE_LOLA_EXACT,
    BASE_AMTFT,
    POLICY_ID_PL0,
    POLICY_ID_PL1,
    META_MINIMUM,
)

payoffs_per_groups = None

prefix = (
    "~/dev-maxime/CLR/vm-data/instance-60-cpu-4-preemtible/meta_game_compare"
)
# prefix = "~/ray_results/meta_game_compare"
prefix = os.path.expanduser(prefix)
META_POLICY_SAVE_PATHS = {
    META_APLHA_RANK: prefix
    + "/2021_05_26/19_16_35/meta_game/meta_policies.json",
    META_APLHA_PURE: prefix
    + "/2021_05_26/19_26_28/meta_game/meta_policies.json",
    META_REPLICATOR_DYNAMIC: prefix
    + "/2021_05_26/19_36_11/meta_game/meta_policies.json",
    META_REPLICATOR_DYNAMIC_ZERO_INIT: prefix
    + "/2021_05_26/19_47_52/meta_game/meta_policies.json",
    META_RANDOM: prefix + "/2021_05_26/19_59_35/meta_game/meta_policies.json",
    META_PG: prefix + "/2021_05_26/20_09_19/meta_game/meta_policies.json",
    META_LOLA_EXACT: prefix
    + "/2021_05_26/20_24_26/meta_game/meta_policies.json",
    META_SOS: prefix + "/2021_05_26/22_10_37/meta_game/meta_policies.json",
    META_UNIFORM: prefix + "/2021_05_26/22_21_07/meta_game/meta_policies.json",
    META_MINIMUM: prefix + "/2021_05_27/19_24_36/meta_game/meta_policies.json",
}


def main(debug, base_game_algo=None, pair_of_meta_game_algo=None):
    """Evaluate meta game performances"""

    n_replicates_over_full_exp = 2 if debug else 20
    train_n_replicates = 1
    n_cross_play = 4
    eval_over_n_epi = 2 if debug else 10
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("meta_game_compare")

    (
        rllib_configs_by_meta_solvers,
        stop_config,
        hp,
        hp_eval,
        trainer,
    ) = _get_rllib_config_by_meta_solvers(
        pair_of_meta_game_algo,
        debug,
        seeds,
        exp_name,
        n_replicates_over_full_exp,
        base_game_algo,
    )
    stop_config["episodes_total"] = eval_over_n_epi
    master_rllib_config = _mix_rllib_config(
        hp_eval, rllib_configs_by_meta_solvers, n_cross_play
    )
    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)
    tune_analysis = various_algo_meta_game._train_with_tune(
        master_rllib_config,
        stop_config,
        hp_eval,
        trainer,
        plot_aggregates=False,
    )
    ray.shutdown()

    various_algo_meta_game._extract_metric_and_log_and_plot(
        tune_analysis,
        hp,
        hp_eval,
        title=f"BASE({base_game_algo}) META({pair_of_meta_game_algo})",
    )


def _get_rllib_config_by_meta_solvers(
    pair_of_meta_game_algo,
    debug,
    seeds,
    exp_name,
    n_replicates_over_full_exp,
    base_game_algo,
):
    rllib_configs_by_meta_solvers = []
    for meta_game_algo in pair_of_meta_game_algo:
        hp = various_algo_meta_game._get_hyperparameters(
            debug, seeds, exp_name, base_game_algo, meta_game_algo
        )
        hp["n_replicates_over_full_exp"] = n_replicates_over_full_exp
        global payoffs_per_groups
        (
            hp["payoff_matrices"],
            hp["actions_possible"],
            hp["base_ckpt_per_replicat"],
            payoffs_per_groups,
        ) = various_algo_meta_game._form_n_matrices_from_base_game_payoffs(hp)
        hp = _load_meta_policies(hp, meta_game_algo)
        rllib_configs, stop_config, trainer, hp_eval = _get_rllib_configs(hp)
        rllib_configs_by_meta_solvers.append(rllib_configs)
    return rllib_configs_by_meta_solvers, stop_config, hp, hp_eval, trainer


def _load_meta_policies(hp, meta_game_algo):
    meta_policy_save_path = META_POLICY_SAVE_PATHS[meta_game_algo]
    with open(meta_policy_save_path) as json_file:
        json_content = json.load(json_file)
    clamped_meta_policies = copy.deepcopy(
        json_content["clamped_meta_policies"]
    )
    clamped_meta_policies = [
        {k: torch.tensor(v) for k, v in el.items()}
        for el in clamped_meta_policies
    ]
    hp["meta_game_policy_distributions"] = clamped_meta_policies

    return hp


def _mix_rllib_config(hp, rllib_configs_by_meta_solvers, n_cross_play):
    all_config_mix = []
    assert n_cross_play % 2 == 0
    n_meta_policies_meta_solver_1 = len(rllib_configs_by_meta_solvers[0])
    for meta_solver_1_idx in range(n_meta_policies_meta_solver_1):
        for player_order in range(2):
            for i in range(n_cross_play // 2):
                meta_solver_1_config = rllib_configs_by_meta_solvers[
                    player_order
                ]
                meta_solver_2_config = rllib_configs_by_meta_solvers[
                    (player_order + 1) % 2
                ]

                meta_solver_2_idx = random.randint(
                    0, len(meta_solver_2_config) - 1
                )

                pl_1_config = meta_solver_1_config[meta_solver_1_idx][
                    "multiagent"
                ]["policies"][POLICY_ID_PL0]
                pl_2_config = meta_solver_2_config[meta_solver_2_idx][
                    "multiagent"
                ]["policies"][POLICY_ID_PL1]
                mix_policies = {
                    POLICY_ID_PL0: copy.deepcopy(pl_1_config),
                    POLICY_ID_PL1: copy.deepcopy(pl_2_config),
                }
                all_config_mix.append(mix_policies)

    master_config = copy.deepcopy(rllib_configs_by_meta_solvers[0][0])
    print("len(all_config_mix)", len(all_config_mix))
    master_config["multiagent"]["policies"] = tune.grid_search(all_config_mix)
    return master_config


def _get_rllib_configs(hp):
    rllib_configs_for_one_meta_solver = []
    for meta_game_idx in range(hp["n_replicates_over_full_exp"]):
        (
            rllib_config,
            stop_config,
            trainer,
            hp_eval,
        ) = various_algo_meta_game._get_final_base_game_rllib_config(
            copy.deepcopy(hp), meta_game_idx
        )

        rllib_configs_for_one_meta_solver.append(rllib_config)
    return rllib_configs_for_one_meta_solver, stop_config, trainer, hp_eval


if __name__ == "__main__":
    debug_mode = True
    loop_over_main = True

    if loop_over_main:
        base_game_algo_to_eval = (
            BASE_LOLA_EXACT,
            # BASE_NEGOTIATION,
        )
        meta_game_algo_to_eval = (
            # META_APLHA_RANK,
            # META_APLHA_PURE,
            # META_REPLICATOR_DYNAMIC,
            # META_REPLICATOR_DYNAMIC_ZERO_INIT,
            META_RANDOM,
            META_PG,
            # META_LOLA_EXACT,
            # META_SOS,
            META_UNIFORM,
            META_MINIMUM,
        )
        pairs_seen = []
        for base_game_algo_ in base_game_algo_to_eval:
            for meta_game_algo_1 in meta_game_algo_to_eval:
                for meta_game_algo_2 in meta_game_algo_to_eval:
                    if meta_game_algo_1 != meta_game_algo_2:
                        meta_pair = [meta_game_algo_1, meta_game_algo_2]
                        if sorted(meta_pair) not in pairs_seen:
                            main(
                                debug_mode,
                                base_game_algo_,
                                meta_pair,
                            )
                            pairs_seen.append(sorted(meta_pair))
                        else:
                            print("skipping pair", meta_pair)
    else:
        main(debug_mode)
