import copy
import os

import numpy as np
import pandas as pd
import ray
import torch
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.pg.pg_torch_policy import PGTorchPolicy, pg_loss_stats
from ray.tune.integration.wandb import WandbLoggerCallback

from marltoolbox import utils
from marltoolbox.algos import population
from marltoolbox.algos.lola.train_exact_tune_class_API import LOLAExactTrainer
from marltoolbox.algos.stochastic_population import StochasticPopulation
from marltoolbox.algos.welfare_coordination import MetaGameSolver
from marltoolbox.envs.matrix_sequential_social_dilemma import (
    TwoPlayersCustomizableMatrixGame,
)
from marltoolbox.examples.rllib_api import pg_ipd
from marltoolbox.experiments.rllib_api import amtft_meta_game
from marltoolbox.experiments.rllib_api import amtft_various_env
from marltoolbox.experiments.tune_class_api import (
    lola_exact_meta_game,
    lola_exact_official,
)
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import (
    log,
    miscellaneous,
    callbacks,
)


def main(debug):
    """Evaluate meta game performances"""

    train_n_replicates = 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("meta_game_compare")

    hparams = _get_hyperparameters(debug, seeds, exp_name)
    (
        hparams["payoff_matrices"],
        hparams["all_welfares_fn"],
        hparams["base_ckpt_per_replicat"],
    ) = form_n_matrices_from_base_payoffs(hparams)

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=hparams["debug"])

    hparams["exp_name"] = os.path.join(hparams["exp_name"], "meta_game")
    hparams["meta_game_policy_distributions"] = _train_meta_policies(hparams)
    hparams["exp_name"] = os.path.split(hparams["exp_name"])[0]
    hparams["exp_name"] = os.path.join(hparams["exp_name"], "final_base_game")
    tune_analysis, hp_eval = _evaluate_in_base_game(hparams)

    ray.shutdown()

    (
        mean_player_1_payoffs,
        mean_player_2_payoffs,
    ) = _extract_metrics(tune_analysis, hparams)
    results = []
    for player1_avg_r_one_replicate, player2_avg_r_one_replicate in zip(
        mean_player_1_payoffs, mean_player_2_payoffs
    ):
        results.append(
            (player1_avg_r_one_replicate, player2_avg_r_one_replicate)
        )
    amtft_meta_game.save_to_json(exp_name=hparams["exp_name"], object=results)
    amtft_meta_game.plot_results(
        exp_name=hparams["exp_name"],
        results=results,
        hp_eval=hp_eval,
        format_fn=_format_result_for_plotting,
        jitter=0.05,
    )


BASE_AMTFT = "amTFT"
BASE_LOLAExact = "LOLA-Exact"
META_LOLAExact = "LOLA-Exact"
META_PG = "PG"
META_APLHA_RANK = "alpha-rank"
META_APLHA_PURE = "alpha-rank pure strategy"
META_REPLICATOR_DYNAMIC = "replicator dynamic"


def _get_hyperparameters(debug, seeds, exp_name):
    hp = {
        "base_game_policies": BASE_AMTFT,
        # "base_game_policies": BASE_LOLAExact,
        #
        # "meta_game_policies": META_PG,
        # "meta_game_policies": META_LOLAExact,
        "meta_game_policies": META_APLHA_RANK,
        # "meta_game_policies": META_APLHA_PURE,
        # "meta_game_policies": META_REPLICATOR_DYNAMIC,
        #
        "players_ids": ["player_row", "player_col"],
        "use_r2d2": True,
    }
    if hp["base_game_policies"] == BASE_AMTFT:
        hp.update(
            amtft_meta_game.get_hyperparameters(
                debug=False, use_r2d2=hp["use_r2d2"]
            )
        )
    elif hp["base_game_policies"] == BASE_LOLAExact:
        hp.update(lola_exact_meta_game.get_hyperparameters(debug=False))
    else:
        raise ValueError()

    hp.update(
        {
            "debug": debug,
            "seeds": seeds,
            "exp_name": exp_name,
            "wandb": {
                "project": "meta_game_compare",
                "group": exp_name,
                "api_key_file": os.path.join(
                    os.path.dirname(__file__), "../../../api_key_wandb"
                ),
            },
        }
    )

    players_ids = ["player_row", "player_col"]
    hp["x_axis_metric"] = f"policy_reward_mean.{players_ids[0]}"
    hp["y_axis_metric"] = f"policy_reward_mean.{players_ids[1]}"
    return hp


def form_n_matrices_from_base_payoffs(hp):
    payoffs_per_groups = _get_payoffs_for_every_group_of_base_game_replicates(
        hp
    )
    # In eval
    if hp["base_game_policies"] == BASE_AMTFT:
        if hp["use_r2d2"]:
            # I removed the change to 100 step in eval when I moved to R2D2
            n_steps_per_epi = 20
        else:
            n_steps_per_epi = 100
    elif hp["base_game_policies"] == BASE_LOLAExact:
        n_steps_per_epi = 1  # in fact 200 but the payoffs are already avg
    else:
        raise ValueError()

    (
        payoffs_matrices,
        all_welfares_fn,
        base_ckpt_per_replicat,
    ) = _aggregate_payoffs_groups_into_matrices(
        payoffs_per_groups, n_steps_per_epi
    )
    return payoffs_matrices, all_welfares_fn, base_ckpt_per_replicat


def _get_payoffs_for_every_group_of_base_game_replicates(hp):
    if hp["base_game_policies"] == BASE_AMTFT:
        module = amtft_meta_game
    elif hp["base_game_policies"] == BASE_LOLAExact:
        module = lola_exact_meta_game
    else:
        raise ValueError()

    payoffs_per_groups = []
    for i in range(hp["n_replicates_over_full_exp"]):
        hp_replicate_i = module._load_base_game_results(
            copy.deepcopy(hp), load_base_replicate_i=i
        )

        all_welfare_pairs_wt_payoffs = (
            module._get_all_welfare_pairs_wt_cross_play_payoffs(
                hp_replicate_i, hp_replicate_i["players_ids"]
            )
        )
        payoffs_per_groups.append(
            (all_welfare_pairs_wt_payoffs, hp_replicate_i)
        )
    return payoffs_per_groups


def _aggregate_payoffs_groups_into_matrices(
    payoffs_per_groups, n_steps_per_epi
):
    payoff_matrices = []
    ckpt_per_replicat = []
    all_welfares_fn = None
    for payoffs_for_one_group, hp_replicat_i in payoffs_per_groups:
        (
            one_payoff_matrice,
            tmp_all_welfares_fn,
        ) = _aggregate_payoffs_in_one_matrix(
            payoffs_for_one_group, n_steps_per_epi
        )
        payoff_matrices.append(one_payoff_matrice)
        ckpt_per_replicat.append(hp_replicat_i["ckpt_per_welfare"])
        if all_welfares_fn is None:
            all_welfares_fn = tmp_all_welfares_fn
        assert len(all_welfares_fn) == len(
            tmp_all_welfares_fn
        ), f"{len(all_welfares_fn)} == {len(tmp_all_welfares_fn)}"
    return payoff_matrices, all_welfares_fn, ckpt_per_replicat


def _aggregate_payoffs_in_one_matrix(payoffs_for_one_group, n_steps_per_epi):
    all_welfares_fn = MetaGameSolver.list_all_welfares_fn(
        payoffs_for_one_group
    )
    all_welfares_fn = sorted(tuple(all_welfares_fn))
    n_welfare_fn = len(all_welfares_fn)
    payoff_matrix = np.empty(
        shape=(n_welfare_fn, n_welfare_fn, 2), dtype=np.float
    )
    for row_i, welfare_player_1 in enumerate(all_welfares_fn):
        for col_i, welfare_player_2 in enumerate(all_welfares_fn):
            welfare_pair_name = (
                MetaGameSolver.from_pair_of_welfare_names_to_key(
                    welfare_player_1, welfare_player_2
                )
            )
            # if welfare_pair_name in payoffs_for_one_group.keys():
            # v1 = payoffs_for_one_group[welfare_pair_name][0] / n_steps_per_epi
            # v2 = payoffs_for_one_group[welfare_pair_name][1] / n_steps_per_epi
            # else:
            #     v1 = -999
            #     v2 = -999
            payoff_matrix[row_i, col_i, 0] = (
                payoffs_for_one_group[welfare_pair_name][0] / n_steps_per_epi
            )
            payoff_matrix[row_i, col_i, 1] = (
                payoffs_for_one_group[welfare_pair_name][1] / n_steps_per_epi
            )
    return payoff_matrix, all_welfares_fn


def _train_meta_policies(hp):
    if hp["meta_game_policies"] == META_PG:
        meta_policies = _train_meta_policy_using_pg(hp)
    elif hp["meta_game_policies"] == META_LOLAExact:
        meta_policies = _train_meta_policy_using_lola_exact(hp)
    elif hp["meta_game_policies"] == META_APLHA_RANK:
        meta_policies = _train_meta_policy_using_alpha_rank(hp)
    elif hp["meta_game_policies"] == META_APLHA_PURE:
        meta_policies = _train_meta_policy_using_alpha_rank(
            hp, pure_strategy=True
        )
    elif hp["meta_game_policies"] == META_REPLICATOR_DYNAMIC:
        meta_policies = _train_meta_policy_using_replicator_dynamic(hp)
    else:
        raise ValueError()

    meta_policies = _clamp_policies_normalize(meta_policies)

    return meta_policies


def _clamp_policies_normalize(meta_policies):
    for i in range(len(meta_policies)):
        for player_key, player_meta_pi in meta_policies[i].items():
            assert not (
                any(player_meta_pi > 1.01) or any(player_meta_pi < -0.01)
            ), f"player_meta_pi {player_meta_pi}"
            player_meta_pi = player_meta_pi / player_meta_pi.sum()
            meta_policies[i][player_key] = player_meta_pi.clamp(
                min=0.0, max=1.0
            )
    return meta_policies


def _train_meta_policy_using_pg(hp):
    rllib_config, stop_config = pg_ipd.get_rllib_config(
        hp["seeds"], hp["debug"]
    )
    rllib_config, stop_config = _modify_rllib_config_for_meta_pg_policy(
        rllib_config, stop_config, hp
    )

    tune_analysis = _train_with_tune(rllib_config, stop_config, hp, PGTrainer)

    return _extract_policy_pg(tune_analysis)


def _extract_policy_pg(tune_analysis):
    policies = []
    for trial in tune_analysis.trials:
        next_act_distrib_idx = 0
        p1_act_distrib = []
        p2_act_distrib = []
        p1_info = trial.last_result["info"]["learner"]["player_row"]
        p2_info = trial.last_result["info"]["learner"]["player_col"]
        prefix = "act_dist_inputs_single_act"
        while True:
            p1_act_distrib.append(p1_info[f"{prefix}{next_act_distrib_idx}"])
            p2_act_distrib.append(p2_info[f"{prefix}{next_act_distrib_idx}"])
            next_act_distrib_idx += 1
            if f"{prefix}{next_act_distrib_idx}" not in p1_info.keys():
                break
        policy_player_1 = torch.softmax(torch.tensor(p1_act_distrib), dim=0)
        policy_player_2 = torch.softmax(torch.tensor(p2_act_distrib), dim=0)
        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
        print("PG meta policy extracted")
        print("policy_player_1 ", policy_player_1)
        print("policy_player_2 ", policy_player_2)
    return policies


def _modify_rllib_config_for_meta_pg_policy(rllib_config, stop_config, hp):
    rllib_config["env"] = TwoPlayersCustomizableMatrixGame
    rllib_config["env_config"]["NUM_ACTIONS"] = len(hp["all_welfares_fn"])
    rllib_config["env_config"]["all_welfares_fn"] = hp["all_welfares_fn"]
    rllib_config["env_config"]["max_steps"] = 1
    rllib_config["model"] = {
        # Number of hidden layers for fully connected net
        "fcnet_hiddens": [64],
        # Nonlinearity for fully connected net (tanh, relu)
        "fcnet_activation": "relu",
    }
    rllib_config["lr"] = 0.003
    stop_config["episodes_total"] = 10 if hp["debug"] else 8000

    rllib_config["env_config"]["linked_data"] = _get_payoff_matrix_grid_search(
        hp
    )
    rllib_config["seed"] = tune.sample_from(
        lambda spec: spec.config["env_config"]["linked_data"][0]
    )
    rllib_config["env_config"]["PAYOFF_MATRIX"] = tune.sample_from(
        lambda spec: spec.config["env_config"]["linked_data"][1]
    )

    rllib_config = _dynamicaly_change_policies_spaces(hp, rllib_config)

    return rllib_config, stop_config


def _dynamicaly_change_policies_spaces(hp, rllib_config):
    MyPGTorchPolicy = PGTorchPolicy.with_updates(
        stats_fn=log.augment_stats_fn_wt_additionnal_logs(
            stats_function=pg_loss_stats
        )
    )

    tmp_env_config = copy.deepcopy(rllib_config["env_config"])
    tmp_env_config["PAYOFF_MATRIX"] = hp["payoff_matrices"][0]
    tmp_env = rllib_config["env"](tmp_env_config)
    for policy_id, policy_config in rllib_config["multiagent"][
        "policies"
    ].items():
        policy_config = list(policy_config)
        policy_config[0] = MyPGTorchPolicy
        policy_config[1] = tmp_env.OBSERVATION_SPACE
        policy_config[2] = tmp_env.ACTION_SPACE
        rllib_config["multiagent"]["policies"][policy_id] = tuple(
            policy_config
        )
    return rllib_config


def _train_meta_policy_using_lola_exact(hp):
    lola_exact_hp = lola_exact_meta_game.get_hyperparameters(hp["debug"])

    tune_config, stop_config, _ = lola_exact_official.get_tune_config(
        lola_exact_hp
    )

    tune_config, stop_config = _modify_tune_config_for_meta_lola_exact(
        hp, tune_config, stop_config
    )

    tune_analysis = _train_with_tune(
        tune_config, stop_config, hp, LOLAExactTrainer
    )
    return _extract_policy_lola_exact(tune_analysis)


def _extract_policy_lola_exact(tune_analysis):
    policies = []
    for trial in tune_analysis.trials:
        policy_player_1 = trial.last_result["policy1"][-1, :]
        policy_player_2 = trial.last_result["policy2"][-1, :]
        policy_player_1 = torch.tensor(policy_player_1)
        policy_player_2 = torch.tensor(policy_player_2)
        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
        print("LOLA-Exact meta policy extracted")
        print("policy_player_1 ", policy_player_1)
        print("policy_player_2 ", policy_player_2)
    return policies


def _train_with_tune(
    rllib_config,
    stop_config,
    hp,
    trainer,
    wandb=True,
    plot_aggregates=True,
):
    tune_analysis = tune.run(
        trainer,
        config=rllib_config,
        stop=stop_config,
        name=hp["exp_name"],
        # log_to_file=False if hp["debug"] else True,
        # callbacks=None
        # if hp["debug"] or not wandb
        # else [
        #     WandbLoggerCallback(
        #         project=hp["wandb"]["project"],
        #         group=hp["wandb"]["group"],
        #         api_key_file=hp["wandb"]["api_key_file"],
        #         log_config=True,
        #     ),
        # ],
    )

    if not hp["debug"] and plot_aggregates:
        aggregate_and_plot_tensorboard_data.add_summary_plots(
            main_path=os.path.join("~/ray_results/", hp["exp_name"]),
            plot_keys=hp["plot_keys"],
            plot_assemble_tags_in_one_plot=hp["plot_assemblage_tags"],
        )
    return tune_analysis


def _modify_tune_config_for_meta_lola_exact(hp, tune_config, stop_config):
    stop_config["episodes_total"] *= tune_config["trace_length"]
    tune_config["re_init_every_n_epi"] *= tune_config["trace_length"]
    tune_config["trace_length"] = 1
    tune_config["env_name"] = "custom_payoff_matrix"
    tune_config["all_welfares_fn"] = hp["all_welfares_fn"]
    tune_config["linked_data"] = _get_payoff_matrix_grid_search(hp)
    tune_config["seed"] = tune.sample_from(
        lambda spec: spec.config["linked_data"][0]
    )
    tune_config["custom_payoff_matrix"] = tune.sample_from(
        lambda spec: spec.config["linked_data"][1]
    )

    return tune_config, stop_config


def _get_payoff_matrix_grid_search(hp):
    # payoff_matrices = [
    #     payoff_matrice_data for payoff_matrice_data in hp["payoff_matrices"]
    # ]
    payoff_matrices = copy.deepcopy(hp["payoff_matrices"])
    seeds = miscellaneous.get_random_seeds(len(payoff_matrices))
    linked_data = [
        (seed, matrix) for seed, matrix in zip(seeds, payoff_matrices)
    ]
    return tune.grid_search(linked_data)


def _evaluate_in_base_game(hp):
    all_rllib_configs = []
    for meta_game_idx in range(hp["n_replicates_over_full_exp"]):
        (
            rllib_config,
            stop_config,
            trainer,
            hp_eval,
        ) = _get_final_base_game_rllib_config(copy.deepcopy(hp), meta_game_idx)

        all_rllib_configs.append(rllib_config)

    master_rllib_config = amtft_meta_game._mix_rllib_config(
        all_rllib_configs, hp_eval=hp
    )
    tune_analysis = _train_with_tune(
        master_rllib_config,
        stop_config,
        hp,
        trainer,
        wandb=False,
        plot_aggregates=False,
    )
    return tune_analysis, hp_eval


def _get_final_base_game_rllib_config(hp, meta_game_idx):
    if hp["base_game_policies"] == BASE_AMTFT:
        (
            stop_config,
            env_config,
            rllib_config,
            trainer,
            hp_eval,
        ) = _get_rllib_config_for_base_amTFT_policy(hp)
    elif hp["base_game_policies"] == BASE_LOLAExact:
        (
            stop_config,
            env_config,
            rllib_config,
            trainer,
            hp_eval,
        ) = _get_rllib_config_for_base_lolaexact_policy(hp)

    rllib_config["multiagent"]["policies_to_train"] = ["None"]
    tmp_env = rllib_config["env"](rllib_config["env_config"])
    policies = rllib_config["multiagent"]["policies"]
    rllib_config["callbacks"] = callbacks.merge_callbacks(
        callbacks.PolicyCallbacks,
        log.get_logging_callbacks_class(
            log_full_epi=True,
            log_full_epi_interval=1,
            log_from_policy=True,
            log_from_policy_in_evaluation=True,
        ),
    )
    rllib_config["seed"] = tune.sample_from(
        lambda spec: miscellaneous.get_random_seeds(1)[0]
    )

    if not hp["debug"]:
        stop_config["episodes_total"] = 10

    for policy_id, policy_config in policies.items():
        policy_config = list(policy_config)

        stochastic_population_policy_config = (
            _create_one_stochastic_population_config(
                hp, meta_game_idx, policy_id, policy_config
            )
        )

        policy_config[0] = StochasticPopulation
        policy_config[1] = tmp_env.OBSERVATION_SPACE
        policy_config[2] = tmp_env.ACTION_SPACE
        policy_config[3] = stochastic_population_policy_config

        rllib_config["multiagent"]["policies"][policy_id] = tuple(
            policy_config
        )

    return rllib_config, stop_config, trainer, hp_eval


def _get_rllib_config_for_base_amTFT_policy(hp):
    hp_eval = amtft_various_env.get_hyperparameters(
        hp["debug"],
        train_n_replicates=1,
        env="IteratedAsymBoS",
        use_r2d2=hp["use_r2d2"],
    )
    hp_eval = amtft_various_env.modify_hyperparams_for_the_selected_env(
        hp_eval
    )

    (
        rllib_config,
        env_config,
        stop_config,
        hp_eval,
    ) = amtft_various_env._generate_eval_config(hp_eval)

    if hp["use_r2d2"]:
        trainer = dqn.r2d2.R2D2Trainer
    else:
        trainer = dqn.dqn.DQNTrainer

    return stop_config, env_config, rllib_config, trainer, hp_eval


def _get_rllib_config_for_base_lolaexact_policy(hp):
    lola_exact_hp = lola_exact_official.get_hyperparameters(
        debug=hp["debug"], env="IteratedAsymBoS", train_n_replicates=1
    )
    (
        hp_eval,
        rllib_config,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    ) = lola_exact_official.generate_eval_config(lola_exact_hp)

    trainer = PGTrainer

    # rllib_config["callbacks"] = callbacks.PolicyCallbacks

    return stop_config, env_config, rllib_config, trainer, lola_exact_hp


def _create_one_stochastic_population_config(
    hp, meta_game_idx, policy_id, policy_config
):
    """
    This policy config is composed of 3 levels:
    The top level: one stochastic population policies per player. This
    policies stochasticly select (given some proba distribution)
    which nested policy to use.
    The intermediary(nested) level: one population (of identical policies) per
    welfare function. This policy selects randomly which policy from its
    population to use.
    The bottom(base) level: amTFT or LOLA-Exact policies used by the
    intermediary level. (amTFT contains another nested level)
    """
    stochastic_population_policy_config = {
        "nested_policies": [],
        "sampling_policy_distribution": hp["meta_game_policy_distributions"][
            meta_game_idx
        ][policy_id],
    }

    print('hp["base_ckpt_per_replicat"]', hp["base_ckpt_per_replicat"])
    print('hp["all_welfares_fn"]', hp["all_welfares_fn"])
    for welfare_i in hp["all_welfares_fn"]:
        one_nested_population_config = _create_one_vanilla_population_config(
            hp,
            policy_id,
            copy.deepcopy(policy_config),
            meta_game_idx,
            welfare_i,
        )

        stochastic_population_policy_config["nested_policies"].append(
            one_nested_population_config
        )

    return stochastic_population_policy_config


def _create_one_vanilla_population_config(
    hp,
    policy_id,
    policy_config,
    meta_game_idx,
    welfare_i,
):
    base_policy_class = copy.deepcopy(policy_config[0])
    base_policy_config = copy.deepcopy(policy_config[3])

    nested_population_config = copy.deepcopy(population.DEFAULT_CONFIG)
    nested_population_config.update(
        {
            "policy_checkpoints": hp["base_ckpt_per_replicat"][meta_game_idx][
                welfare_i
            ],
            "nested_policies": [
                {
                    "Policy_class": base_policy_class,
                    "config_update": base_policy_config,
                }
            ],
            "policy_id_to_load": policy_id,
        }
    )

    intermediary_config = {
        "Policy_class": population.PopulationOfIdenticalAlgo,
        "config_update": nested_population_config,
    }

    return intermediary_config


def _extract_metrics(tune_analysis, hp_eval):
    player_1_payoffs = utils.tune_analysis.extract_value_from_last_training_iteration_for_each_trials(
        tune_analysis, metric=hp_eval["x_axis_metric"]
    )
    player_2_payoffs = utils.tune_analysis.extract_value_from_last_training_iteration_for_each_trials(
        tune_analysis, metric=hp_eval["y_axis_metric"]
    )
    print("player_1_payoffs", player_1_payoffs)
    print("player_2_payoffs", player_2_payoffs)
    return player_1_payoffs, player_2_payoffs


def _format_result_for_plotting(results):
    data_groups_per_mode = {}
    df_rows = []
    for player1_avg_r_one_replicate, player2_avg_r_one_replicate in results:
        df_row_dict = {
            "": (
                player1_avg_r_one_replicate,
                player2_avg_r_one_replicate,
            )
        }
        df_rows.append(df_row_dict)
    data_groups_per_mode["cross-play"] = pd.DataFrame(df_rows)
    return data_groups_per_mode


def _train_meta_policy_using_alpha_rank(hp, pure_strategy=False):
    from open_spiel.python.egt import alpharank
    from open_spiel.python.algorithms.psro_v2 import utils as psro_v2_utils

    payoff_matrices = copy.deepcopy(hp["payoff_matrices"])

    policies = []
    for payoff_matrix in payoff_matrices:

        payoff_tables_per_player = [
            payoff_matrix[:, :, 0],
            payoff_matrix[:, :, 1],
        ]

        joint_arank = alpharank.sweep_pi_vs_alpha(payoff_tables_per_player)
        (
            policy_player_1,
            policy_player_2,
        ) = psro_v2_utils.get_alpharank_marginals(
            payoff_tables_per_player, joint_arank
        )
        policy_player_1 = torch.tensor(policy_player_1)
        policy_player_2 = torch.tensor(policy_player_2)
        if pure_strategy:
            policy_player_1 = policy_player_1 == policy_player_1.max()
            policy_player_2 = policy_player_2 == policy_player_2.max()
            policy_player_1 = policy_player_1.float()
            policy_player_2 = policy_player_2.float()

        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
    print("alpha rank meta policies", policies)
    return policies


def _train_meta_policy_using_replicator_dynamic(hp):
    from open_spiel.python.algorithms.projected_replicator_dynamics import (
        projected_replicator_dynamics,
    )

    payoff_matrices = copy.deepcopy(hp["payoff_matrices"])
    policies = []
    for payoff_matrix in payoff_matrices:
        payoff_tables_per_player = [
            payoff_matrix[:, :, 0],
            payoff_matrix[:, :, 1],
        ]
        policy_player_1, policy_player_2 = projected_replicator_dynamics(
            payoff_tables_per_player, prd_gamma=0.0
        )

        policy_player_1 = torch.tensor(policy_player_1)
        policy_player_2 = torch.tensor(policy_player_2)
        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
    print("replicator dynamic meta policies", policies)
    return policies


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
