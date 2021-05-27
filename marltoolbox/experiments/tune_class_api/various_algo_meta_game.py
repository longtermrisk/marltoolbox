import copy
import os
import pickle
import random

import numpy as np
import pandas as pd
import ray
import torch
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.pg.pg_torch_policy import PGTorchPolicy, pg_loss_stats

from marltoolbox.algos import population, welfare_coordination
from marltoolbox.algos.lola.train_exact_tune_class_API import LOLAExactTrainer
from marltoolbox.algos.sos import SOSTrainer
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
from marltoolbox.experiments.tune_class_api import sos_exact_official
from marltoolbox.scripts import (
    aggregate_and_plot_tensorboard_data,
    plot_meta_policies,
)
from marltoolbox.utils import log, miscellaneous, callbacks, exp_analysis, path

EPSILON = 1e-6
POLICY_ID_PL0 = "player_row"
POLICY_ID_PL1 = "player_col"


def main(debug, base_game_algo=None, meta_game_algo=None):
    """Evaluate meta game performances"""

    train_n_replicates = 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("meta_game_compare")

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)
    hparams = _get_hyperparameters(
        debug, seeds, exp_name, base_game_algo, meta_game_algo
    )
    (
        hparams["payoff_matrices"],
        hparams["actions_possible"],
        hparams["base_ckpt_per_replicat"],
    ) = _form_n_matrices_from_base_game_payoffs(hparams)
    hparams["meta_game_policy_distributions"] = _train_meta_policies(hparams)
    tune_analysis, hp_eval = _evaluate_in_base_game(hparams)
    ray.shutdown()

    _extract_metric_and_log_and_plot(tune_analysis, hparams, hp_eval)


def _extract_metric_and_log_and_plot(tune_analysis, hparams, hp_eval):
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
    result_to_json = {"results": copy.deepcopy(results)}
    coordination_success = _extract_coordination_metric(tune_analysis)
    print("coordination_success", coordination_success)
    result_to_json["coordination_success"] = np.array(
        coordination_success
    ).tolist()
    result_to_json["mean_coordination_success"] = np.array(
        coordination_success
    ).mean()

    amtft_meta_game.save_to_json(
        exp_name=hparams["exp_name"], object=result_to_json
    )
    amtft_meta_game.plot_results(
        exp_name=hparams["exp_name"],
        results=results,
        hp_eval=hp_eval,
        format_fn=_format_result_for_plotting,
        jitter=0.05,
    )


BASE_AMTFT = "amTFT"
BASE_LOLA_EXACT = "base LOLA-Exact"
BASE_NEGOTIATION = "base negociation"
META_LOLA_EXACT = "meta LOLA-Exact"
META_PG = "PG"
META_SOS = "SOS"
META_APLHA_RANK = "alpha-rank"
META_APLHA_PURE = "alpha-rank pure strategy"
META_REPLICATOR_DYNAMIC = "replicator dynamic"
META_REPLICATOR_DYNAMIC_ZERO_INIT = "replicator dynamic with zero init"
META_RANDOM = "Random"
META_UNIFORM = "Robustness tau=0.0"


def _get_hyperparameters(
    debug, seeds, exp_name, base_game_algo=None, meta_game_algo=None
):
    hp = {
        # "base_game_policies": BASE_AMTFT,
        # "base_game_policies": BASE_LOLA_EXACT,
        "base_game_policies": BASE_NEGOTIATION,
        #
        # "meta_game_policies": META_PG,
        # "meta_game_policies": META_LOLA_EXACT,
        # "meta_game_policies": META_APLHA_RANK,
        # "meta_game_policies": META_APLHA_PURE,
        # "meta_game_policies": META_REPLICATOR_DYNAMIC,
        # "meta_game_policies": META_REPLICATOR_DYNAMIC_ZERO_INIT,
        "meta_game_policies": META_RANDOM,
        # "meta_game_policies": META_UNIFORM,
        #
        "apply_announcement_protocol": True,
        "negotitation_process": 2,
        #
        "players_ids": ["player_row", "player_col"],
        "use_r2d2": True,
    }

    if base_game_algo is not None:
        hp["base_game_policies"] = base_game_algo
    if meta_game_algo is not None:
        hp["meta_game_policies"] = meta_game_algo

    hp["load_meta_game_payoff_matrices"] = (
        hp["base_game_policies"] == BASE_NEGOTIATION
    )
    hp["evaluate_meta_policies_reading_meta_game_payoff_matrices"] = (
        hp["base_game_policies"] == BASE_NEGOTIATION
    )

    if hp["load_meta_game_payoff_matrices"]:
        assert hp["base_game_policies"] == BASE_NEGOTIATION
    if hp["evaluate_meta_policies_reading_meta_game_payoff_matrices"]:
        assert hp["base_game_policies"] == BASE_NEGOTIATION

    if hp["base_game_policies"] == BASE_AMTFT:
        hp.update(
            amtft_meta_game.get_hyperparameters(
                debug=debug, use_r2d2=hp["use_r2d2"]
            )
        )
    elif hp["base_game_policies"] == BASE_LOLA_EXACT:
        hp.update(lola_exact_meta_game.get_hyperparameters(debug=debug))
    elif hp["base_game_policies"] == BASE_NEGOTIATION:
        hp.update(_get_negociation_hyperparameters(debug=debug))
        assert hp["evaluate_meta_policies_reading_meta_game_payoff_matrices"]
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
    if hp["base_game_policies"] == BASE_NEGOTIATION:
        hp["x_axis_metric"] = f"policy_reward_mean.{players_ids[0]}"
        hp["y_axis_metric"] = f"policy_reward_mean.{players_ids[1]}"
    else:
        hp["x_axis_metric"] = f"policy_reward_mean/{players_ids[0]}"
        hp["y_axis_metric"] = f"policy_reward_mean/{players_ids[1]}"

    return hp


payoffs_per_groups = None


def _get_negociation_hyperparameters(debug):
    hp = {
        "n_replicates_over_full_exp": 2 if debug else 40,
        "n_self_play_in_final_meta_game": 0,
        "n_cross_play_in_final_meta_game": 1 if debug else 10,
        "env_name": "Nogetiation",
        "x_limits": (0.0, 1.0),
        "y_limits": (0.0, 1.0),
        "plot_axis_scale_multipliers": (1, 1),
        "plot_keys": aggregate_and_plot_tensorboard_data.PLOT_KEYS,
        "plot_assemblage_tags": aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS,
        "data_prefix": (
            "/home/maxime/ssd_space/CLR/marltoolbox/marltoolbox/experiments"
            "/tune_class_api/"
        ),
        # "data_prefix": (
        #         "/home/maxime-riche/marltoolbox/marltoolbox/experiments"
        #         "/tune_class_api/"
        # ),
    }
    return hp


def _form_n_matrices_from_base_game_payoffs(hp):
    global payoffs_per_groups
    if hp["load_meta_game_payoff_matrices"]:
        (
            payoffs_matrices,
            actions_possible,
            base_ckpt_per_replicat,
            payoffs_per_groups,
        ) = _load_payoffs_matrices(hp)
        return (
            payoffs_matrices,
            actions_possible,
            base_ckpt_per_replicat,
        )
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
    elif hp["base_game_policies"] == BASE_LOLA_EXACT:
        if hp["debug"]:
            n_steps_per_epi = 40
        else:
            n_steps_per_epi = 1  # in fact 200 but the payoffs are already avg
    else:
        raise ValueError()

    if hp["apply_announcement_protocol"]:
        (
            payoffs_matrices,
            actions_possible,
            base_ckpt_per_replicat,
        ) = _aggregate_payoffs_groups_into_matrices_wt_announcement_protocol(
            payoffs_per_groups, n_steps_per_epi, hp
        )
    else:
        (
            payoffs_matrices,
            actions_possible,
            base_ckpt_per_replicat,
        ) = _aggregate_payoffs_groups_into_matrices(
            payoffs_per_groups, n_steps_per_epi, hp
        )

    assert len(payoffs_matrices) == hp["n_replicates_over_full_exp"]
    return payoffs_matrices, actions_possible, base_ckpt_per_replicat


def _get_payoffs_for_every_group_of_base_game_replicates(hp):
    if hp["base_game_policies"] == BASE_AMTFT:
        module = amtft_meta_game
    elif hp["base_game_policies"] == BASE_LOLA_EXACT:
        module = lola_exact_meta_game
    else:
        raise ValueError(f'base_game_policies {hp["base_game_policies"]}')

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


def _aggregate_payoffs_groups_into_matrices_wt_announcement_protocol(
    payoffs_per_groups, n_steps_per_epi, hp
):
    payoffs_matrices = []
    ckpt_per_replicat = []
    previous_welfare_fn_sets = None
    for i, (payoffs_for_one_group, hp_replicat_i) in enumerate(
        payoffs_per_groups
    ):
        ckpt_per_replicat.append(hp_replicat_i["ckpt_per_welfare"])

        announcement_protocol_solver_p1 = welfare_coordination.MetaGameSolver()
        announcement_protocol_solver_p1.setup_meta_game(
            payoffs_per_groups[i][0],
            own_player_idx=0,
            opp_player_idx=1,
            own_default_welfare_fn="utilitarian",
            opp_default_welfare_fn="inequity aversion"
            if hp["base_game_policies"] == BASE_AMTFT
            else "egalitarian",
        )

        welfare_fn_sets = announcement_protocol_solver_p1.welfare_fn_sets
        if previous_welfare_fn_sets is not None:
            assert welfare_fn_sets == previous_welfare_fn_sets
        previous_welfare_fn_sets = welfare_fn_sets
        print("\nwelfare_fn_sets", welfare_fn_sets)
        n_set_of_welfare_sets = len(welfare_fn_sets)
        payoff_matrix = np.empty(
            shape=(n_set_of_welfare_sets, n_set_of_welfare_sets, 2),
            dtype=np.float,
        )
        for own_welfare_set_idx, own_welfare_set_announced in enumerate(
            welfare_fn_sets
        ):
            for opp_welfare_set_idx, opp_wefare_set in enumerate(
                welfare_fn_sets
            ):
                cell_payoffs = (
                    announcement_protocol_solver_p1._compute_meta_payoff(
                        own_welfare_set_announced, opp_wefare_set
                    )
                )
                payoff_matrix[own_welfare_set_idx, opp_welfare_set_idx, 0] = (
                    cell_payoffs[0] / n_steps_per_epi
                )
                payoff_matrix[own_welfare_set_idx, opp_welfare_set_idx, 1] = (
                    cell_payoffs[1] / n_steps_per_epi
                )
        amtft_meta_game.save_to_json(
            exp_name=hp["exp_name"],
            object={
                "welfare_fn_sets": str(welfare_fn_sets),
                "payoff_matrix": payoff_matrix.tolist(),
            },
            filename=f"payoffs_matrices_{i}.json",
        )
        payoffs_matrices.append(payoff_matrix)
    return payoffs_matrices, welfare_fn_sets, ckpt_per_replicat


def _aggregate_payoffs_groups_into_matrices(
    payoffs_per_groups, n_steps_per_epi, hp
):
    payoff_matrices = []
    ckpt_per_replicat = []
    all_welfares_fn = None
    for i, (payoffs_for_one_group, hp_replicat_i) in enumerate(
        payoffs_per_groups
    ):
        (
            one_payoff_matrice,
            tmp_all_welfares_fn,
        ) = _aggregate_payoffs_in_one_matrix(
            payoffs_for_one_group, n_steps_per_epi
        )
        amtft_meta_game.save_to_json(
            exp_name=hp["exp_name"],
            object=one_payoff_matrice.tolist(),
            filename=f"payoffs_matrices_{i}.json",
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
            payoff_matrix[row_i, col_i, 0] = (
                payoffs_for_one_group[welfare_pair_name][0] / n_steps_per_epi
            )
            payoff_matrix[row_i, col_i, 1] = (
                payoffs_for_one_group[welfare_pair_name][1] / n_steps_per_epi
            )
    return payoff_matrix, all_welfares_fn


def _load_payoffs_matrices(hp):
    if hp["base_game_policies"] == BASE_NEGOTIATION:
        if hp["negotitation_process"] == 1:
            return _load_payoffs_matrices_negotiation_process1(hp)
        elif hp["negotitation_process"] == 2:
            return _load_payoffs_matrices_negotiation_process2(hp)
    else:
        raise NotImplementedError()


def _load_payoffs_matrices_negotiation_process1(hp):
    file_path = os.path.join(
        hp["data_prefix"], "negociation_game_replicates_list.pickle"
    )
    with open(file_path, "rb") as f:
        content = pickle.load(f)

    principal0_replicates = content[:20]
    hp["principal0_replicates"] = principal0_replicates
    payoffs_matrices = []
    n_actions = None
    for i, (mat_pl0, mat_pl1, _, _) in enumerate(principal0_replicates):
        if n_actions is None:
            n_actions = mat_pl0.shape[0]
        assert n_actions == mat_pl0.shape[0] == mat_pl0.shape[1]
        assert n_actions == mat_pl1.shape[0] == mat_pl1.shape[1]
        payoffs_matrices.append(np.stack([mat_pl0, mat_pl1], axis=-1))
    print("payoffs_matrices[0].shape", payoffs_matrices[0].shape)

    actions_possible = []
    for i in range(n_actions):
        actions_possible.append(f"meta_action_{str(i)}")

    base_ckpt_per_replicat = None
    payoffs_per_groups = None

    return (
        payoffs_matrices,
        actions_possible,
        base_ckpt_per_replicat,
        payoffs_per_groups,
    )


def _load_payoffs_matrices_negotiation_process2(hp):
    file_path = os.path.join(
        hp["data_prefix"], "bootstrapped_replicates_prosociality_coeff_0.3"
    )
    with open(file_path, "rb") as f:
        content = pickle.load(f)
    load_from_n_principals = 2
    load_n_bootstrapped_by_principals = (
        hp["n_replicates_over_full_exp"] // load_from_n_principals
    )
    print("load_from_n_principals", load_from_n_principals)
    print(
        "load_n_bootstrapped_by_principals", load_n_bootstrapped_by_principals
    )
    all_principal_replicates = []
    meta_game_matrix_pos_to_principal_idx = []
    for principal_i in range(load_from_n_principals):
        principal_i_replicates = content[principal_i][
            :load_n_bootstrapped_by_principals
        ]
        hp[f"principal{principal_i}_replicates"] = principal_i_replicates
        all_principal_replicates.extend(principal_i_replicates)
        meta_game_matrix_pos_to_principal_idx.extend(
            [principal_i] * len(principal_i_replicates)
        )
    hp[
        "meta_game_matrix_pos_to_principal_idx"
    ] = meta_game_matrix_pos_to_principal_idx
    hp["load_n_bootstrapped_by_principals"] = load_n_bootstrapped_by_principals
    payoffs_matrices = []
    n_actions = None
    for i, (mat_pl0, _, _) in enumerate(all_principal_replicates):
        if n_actions is None:
            n_actions = mat_pl0.shape[0]
        assert n_actions == mat_pl0.shape[0] == mat_pl0.shape[1]
        payoffs_matrices.append(mat_pl0)
    # print("payoffs_matrices[0].shape", payoffs_matrices[0].shape)

    actions_possible = []
    for i in range(n_actions):
        actions_possible.append(f"meta_action_{str(i)}")

    base_ckpt_per_replicat = None
    payoffs_per_groups = None

    return (
        payoffs_matrices,
        actions_possible,
        base_ckpt_per_replicat,
        payoffs_per_groups,
    )


def _train_meta_policies(hp):
    hp["exp_name"] = os.path.join(hp["exp_name"], "meta_game")

    if hp["meta_game_policies"] == META_PG:
        meta_policies = _train_meta_policy_using_pg(hp)
    elif hp["meta_game_policies"] == META_LOLA_EXACT:
        meta_policies = _train_meta_policy_using_lola_exact(hp)
    elif hp["meta_game_policies"] == META_APLHA_RANK:
        meta_policies = _train_meta_policy_using_alpha_rank(hp)
    elif hp["meta_game_policies"] == META_APLHA_PURE:
        meta_policies = _train_meta_policy_using_alpha_rank(
            hp, pure_strategy=True
        )
    elif (
        hp["meta_game_policies"] == META_REPLICATOR_DYNAMIC
        or hp["meta_game_policies"] == META_REPLICATOR_DYNAMIC_ZERO_INIT
    ):
        meta_policies = _train_meta_policy_using_replicator_dynamic(hp)
    elif hp["meta_game_policies"] == META_RANDOM:
        meta_policies = _get_random_meta_policy(hp)
    elif hp["meta_game_policies"] == META_SOS:
        meta_policies = _train_meta_policy_using_sos_exact(hp)
    elif hp["meta_game_policies"] == META_UNIFORM:
        meta_policies = _train_meta_policy_using_robustness(hp)
    else:
        raise ValueError()

    if hp["base_game_policies"] == BASE_NEGOTIATION:
        if hp["meta_game_policies"] != META_RANDOM:
            meta_policies = (
                _convert_negociation_meta_policies_to_original_space(
                    meta_policies, hp
                )
            )

    clamped_meta_policies = _clamp_policies_normalize(meta_policies)

    amtft_meta_game.save_to_json(
        exp_name=hp["exp_name"],
        object={
            "clamped_meta_policies": _convert_to_list(clamped_meta_policies),
            "meta_policies": _convert_to_list(meta_policies),
        },
        filename=f"meta_policies.json",
    )

    exp_dir_path = path.get_exp_dir_from_exp_name(hp["exp_name"])
    plot_meta_policies.plot_policies(
        _convert_to_list(meta_policies),
        hp["actions_possible"],
        title=f'META({hp["meta_game_policies"]}) BASE('
        f'{hp["base_game_policies"]})',
        path_prefix=exp_dir_path + "/",
        announcement_protocol=hp["apply_announcement_protocol"]
        and hp["base_game_policies"] != BASE_NEGOTIATION,
    )

    return clamped_meta_policies


def _convert_negociation_meta_policies_to_original_space(meta_policies, hp):
    meta_policies_in_original_space = []
    for meta_pi_idx, meta_policy in enumerate(meta_policies):
        policy_player_1 = _order_meta_game_policy(
            meta_policy["player_row"], hp, meta_pi_idx, POLICY_ID_PL0
        )
        policy_player_2 = _order_meta_game_policy(
            meta_policy["player_col"], hp, meta_pi_idx, POLICY_ID_PL1
        )
        policy_player_1 = torch.tensor(policy_player_1)
        policy_player_2 = torch.tensor(policy_player_2)
        meta_policy_in_original_space = {
            "player_row": policy_player_1,
            "player_col": policy_player_2,
        }
        meta_policies_in_original_space.append(meta_policy_in_original_space)
    return meta_policies_in_original_space


def _order_meta_game_policy(meta_pi, hp, meta_policy_idx, player_id):
    """Assemble policies to fit the order in the original pmeta game payoff
    matrix (not the bootstrapepd one)"""
    if hp["negotitation_process"] == 1:
        return _order_meta_game_policy_process1(
            meta_pi, hp, meta_policy_idx, player_id
        )
    elif hp["negotitation_process"] == 2:
        return _order_meta_game_policy_process2(
            meta_pi, hp, meta_policy_idx, player_id
        )


def _order_meta_game_policy_process1(meta_pi, hp, meta_policy_idx, player_id):
    _, x_indices, y_indices = hp["principal0_replicates"][meta_policy_idx]
    if player_id == "player_row":
        indices = x_indices
    elif player_id == "player_col":
        indices = y_indices
    else:
        raise ValueError()
    meta_pi_original_space = np.zeros_like(meta_pi)
    for i, val in enumerate(meta_pi):
        original_index = indices[i]
        meta_pi_original_space[original_index] += val

    return meta_pi_original_space


def _order_meta_game_policy_process2(meta_pi, hp, meta_policy_idx, player_id):
    principal_idx = hp["meta_game_matrix_pos_to_principal_idx"][
        meta_policy_idx
    ]
    idx_in_principal = (
        meta_policy_idx % hp["load_n_bootstrapped_by_principals"]
    )
    _, x_indices, y_indices = hp[f"principal{principal_idx}_replicates"][
        idx_in_principal
    ]
    print("principal_idx", principal_idx)
    print("idx_in_principal", idx_in_principal)
    if player_id == "player_row":
        indices = x_indices
    elif player_id == "player_col":
        indices = y_indices
    else:
        raise ValueError()
    meta_pi_original_space = np.zeros_like(meta_pi)
    for i, val in enumerate(meta_pi):
        original_index = indices[i]
        meta_pi_original_space[original_index] += val

    return meta_pi_original_space


def _convert_to_list(list_dict_tensors):
    return [
        {k: v.tolist() for k, v in dict_.items()}
        for dict_ in list_dict_tensors
    ]


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
    rllib_config["env_config"]["NUM_ACTIONS"] = len(hp["actions_possible"])
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


def _train_meta_policy_using_sos_exact(hp):
    lola_exact_hp = sos_exact_official.get_hyperparameters(hp["debug"])

    tune_config, stop_config, _ = sos_exact_official.get_tune_config(
        lola_exact_hp
    )

    tune_config, stop_config = _modify_tune_config_for_meta_sos_exact(
        hp, tune_config, stop_config
    )

    tune_analysis = _train_with_tune(tune_config, stop_config, hp, SOSTrainer)
    return _extract_policy_sos_exact(tune_analysis)


def _modify_tune_config_for_meta_sos_exact(hp, tune_config, stop_config):
    tune_config["env_name"] = None
    tune_config["method"] = "sos"
    tune_config["linked_data"] = _get_payoff_matrix_grid_search(hp)
    tune_config["seed"] = tune.sample_from(
        lambda spec: spec.config["linked_data"][0]
    )
    tune_config["custom_payoff_matrix"] = tune.sample_from(
        lambda spec: spec.config["linked_data"][1]
    )

    return tune_config, stop_config


def _extract_policy_sos_exact(tune_analysis):
    policies = []
    for trial in tune_analysis.trials:
        policy_player_1 = trial.last_result["policy1"][0, :]
        policy_player_2 = trial.last_result["policy2"][0, :]
        policy_player_1 = torch.tensor(policy_player_1)
        policy_player_2 = torch.tensor(policy_player_2)
        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
        print("SOS-Exact meta policy extracted")
        print("policy_player_1 ", policy_player_1)
        print("policy_player_2 ", policy_player_2)
    return policies


def _train_with_tune(
    rllib_config,
    stop_config,
    hp,
    trainer,
    plot_aggregates=True,
):
    tune_analysis = tune.run(
        trainer,
        config=rllib_config,
        stop=stop_config,
        name=hp["exp_name"],
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
    tune_config["linked_data"] = _get_payoff_matrix_grid_search(hp)
    tune_config["seed"] = tune.sample_from(
        lambda spec: spec.config["linked_data"][0]
    )
    tune_config["custom_payoff_matrix"] = tune.sample_from(
        lambda spec: spec.config["linked_data"][1]
    )

    return tune_config, stop_config


def _get_payoff_matrix_grid_search(hp):
    payoff_matrices = copy.deepcopy(hp["payoff_matrices"])
    seeds = miscellaneous.get_random_seeds(len(payoff_matrices))
    linked_data = [
        (seed, matrix) for seed, matrix in zip(seeds, payoff_matrices)
    ]
    return tune.grid_search(linked_data)


def _evaluate_in_base_game(hp):
    hp["exp_name"] = os.path.join(hp["exp_name"], "final_base_game")
    assert hp["n_replicates_over_full_exp"] > 0

    if hp["evaluate_meta_policies_reading_meta_game_payoff_matrices"]:
        if hp["base_game_policies"] == BASE_NEGOTIATION:
            if hp["negotitation_process"] == 1:
                return _evaluate_by_reading_meta_payoff_matrices_process1(hp)
            elif hp["negotitation_process"] == 2:
                return _evaluate_by_reading_meta_payoff_matrices_process2(hp)
        else:
            raise NotImplementedError()
    else:
        return _evaluate_by_playing_in_base_game(hp)


ORIGNAL_NEGOTIATION_PAYOFFS = np.stack(
    [
        np.array(
            [
                [
                    0.73076171,
                    0.72901064,
                    0.73216635,
                    0.66135818,
                    0.56392801,
                    0.46451038,
                    0.45376301,
                ],
                [
                    0.38869923,
                    0.38534233,
                    0.37598014,
                    0.73307645,
                    0.72625536,
                    0.73916543,
                    0.72769892,
                ],
                [
                    0.7302742,
                    0.73917222,
                    0.72813082,
                    0.66200578,
                    0.66214889,
                    0.56320196,
                    0.56070906,
                ],
                [
                    0.46352547,
                    0.45852849,
                    0.45295322,
                    0.39153525,
                    0.38613006,
                    0.38181722,
                    0.37838927,
                ],
                [
                    0.64827693,
                    0.65012288,
                    0.6471073,
                    0.6198647,
                    0.57871825,
                    0.51521456,
                    0.51886076,
                ],
                [
                    0.47413245,
                    0.59999102,
                    0.59605861,
                    0.59867841,
                    0.59904635,
                    0.57479459,
                    0.54586047,
                ],
                [
                    0.50140649,
                    0.47044769,
                    0.46821448,
                    0.57101083,
                    0.56979507,
                    0.57035708,
                    0.57534903,
                ],
            ]
        ),
        np.array(
            [
                [
                    0.30082142,
                    0.29890773,
                    0.3002491,
                    0.30846024,
                    0.30522716,
                    0.29359573,
                    0.28792399,
                ],
                [
                    0.27443337,
                    0.27473691,
                    0.27099788,
                    0.29374766,
                    0.29737911,
                    0.29684028,
                    0.30397776,
                ],
                [
                    0.30072445,
                    0.29653731,
                    0.3035537,
                    0.3089233,
                    0.31072533,
                    0.30742073,
                    0.30733863,
                ],
                [
                    0.28808233,
                    0.29138255,
                    0.29195258,
                    0.27400267,
                    0.27413243,
                    0.26937005,
                    0.26969171,
                ],
                [
                    0.38675746,
                    0.38473225,
                    0.39369282,
                    0.40118337,
                    0.4028841,
                    0.39932922,
                    0.40290347,
                ],
                [
                    0.3954556,
                    0.39829046,
                    0.39688903,
                    0.39803576,
                    0.39812419,
                    0.41143295,
                    0.41318333,
                ],
                [
                    0.41149494,
                    0.41419694,
                    0.41467997,
                    0.4051294,
                    0.39407712,
                    0.40370235,
                    0.40058476,
                ],
            ]
        ),
    ],
    axis=-1,
)


def _evaluate_by_reading_meta_payoff_matrices_process1(hp):
    all_meta_games_idx = list(range(hp["n_replicates_over_full_exp"]))
    trials_results = []
    for _ in all_meta_games_idx:
        # TODO don't use this ppayoff table
        payoff_matrix = copy.deepcopy(ORIGNAL_NEGOTIATION_PAYOFFS)
        meta_games_idx_available = copy.deepcopy(all_meta_games_idx)

        for cross_play_idx in range(hp["n_cross_play_in_final_meta_game"]):
            (
                meta_pi_pl0,
                meta_pi_pl1,
                meta_pi_pl0_idx,
                meta_pi_pl1_idx,
            ) = _select_cross_play_meta_policies_for_payoff_reading(
                hp, meta_games_idx_available
            )

            min_, max_ = payoff_matrix.min(), payoff_matrix.max()
            joint_proba = _compute_joint_meta_pi_proba(
                meta_pi_pl0, meta_pi_pl1
            )
            weighted_avg = payoff_matrix * joint_proba
            payoff_per_player = np.sum(np.sum(weighted_avg, axis=0), axis=0)
            assert np.all(
                payoff_per_player - max_ < EPSILON
            ), f"{payoff_per_player - max_}"
            assert np.all(
                payoff_per_player - min_ > -EPSILON
            ), f"{payoff_per_player - min_}"
            trials_results.append(
                {
                    "policy_reward_mean.player_row": payoff_per_player[0],
                    "policy_reward_mean.player_col": payoff_per_player[1],
                }
            )

    fake_experiment_analysis = (
        exp_analysis.create_fake_experiment_analysis_wt_metrics_only(
            trials_results
        )
    )
    return fake_experiment_analysis, hp


def _evaluate_by_reading_meta_payoff_matrices_process2(hp):
    file_path = os.path.join(
        hp["data_prefix"], "empirical_game_matrices_prosociality_coeff_0.3"
    )
    with open(file_path, "rb") as f:
        payoff_matrix_35x35 = pickle.load(f)

    all_meta_games_idx = list(range(hp["n_replicates_over_full_exp"]))
    trials_results = []
    for meta_games_idx in all_meta_games_idx:
        print("meta_games_idx", meta_games_idx)
        (
            meta_pi_pl0,
            meta_games_idx_available_pl1,
            pl0_principal_i,
        ) = _load_pi_pl_0_and_available_pi_pl1(
            hp, meta_games_idx, all_meta_games_idx
        )

        for cross_play_idx in range(hp["n_cross_play_in_final_meta_game"]):
            payoff_matrix, meta_pi_pl1 = _load_pi_pl1_and_payoff_matrix(
                hp,
                meta_games_idx_available_pl1,
                pl0_principal_i,
                payoff_matrix_35x35,
            )

            min_, max_ = payoff_matrix.min(), payoff_matrix.max()
            joint_proba = _compute_joint_meta_pi_proba(
                meta_pi_pl0, meta_pi_pl1
            )
            weighted_avg = payoff_matrix * joint_proba
            payoff_per_player = np.sum(np.sum(weighted_avg, axis=0), axis=0)
            assert np.all(
                payoff_per_player - max_ < EPSILON
            ), f"{payoff_per_player - max_}"
            assert np.all(
                payoff_per_player - min_ > -EPSILON
            ), f"{payoff_per_player - min_}"
            trials_results.append(
                {
                    "policy_reward_mean.player_row": payoff_per_player[0],
                    "policy_reward_mean.player_col": payoff_per_player[1],
                }
            )

    fake_experiment_analysis = (
        exp_analysis.create_fake_experiment_analysis_wt_metrics_only(
            trials_results
        )
    )
    return fake_experiment_analysis, hp


def _load_pi_pl_0_and_available_pi_pl1(hp, meta_games_idx, all_meta_games_idx):
    meta_pi_pl0_idx = meta_games_idx
    pl0_principal_i = hp["meta_game_matrix_pos_to_principal_idx"][
        meta_pi_pl0_idx
    ]
    meta_pi_pl0 = hp["meta_game_policy_distributions"][meta_pi_pl0_idx][
        POLICY_ID_PL0
    ]
    print(
        "meta_games_idx",
        meta_games_idx,
        "pl0_principal_i",
        pl0_principal_i,
    )
    assert pl0_principal_i == (
        meta_games_idx // hp["load_n_bootstrapped_by_principals"]
    )
    meta_games_idx_available_pl1 = copy.deepcopy(all_meta_games_idx)
    for meta_policy_i, principal_j in enumerate(
        hp["meta_game_matrix_pos_to_principal_idx"]
    ):
        if principal_j == pl0_principal_i:
            meta_games_idx_available_pl1.remove(meta_policy_i)
    print("meta_games_idx_available_pl1", meta_games_idx_available_pl1)
    assert (
        len(meta_games_idx_available_pl1)
        == len(all_meta_games_idx) - hp["load_n_bootstrapped_by_principals"]
    )
    return meta_pi_pl0, meta_games_idx_available_pl1, pl0_principal_i


def _load_pi_pl1_and_payoff_matrix(
    hp, meta_games_idx_available_pl1, pl0_principal_i, payoff_matrix_35x35
):
    meta_pi_pl1_idx = random.choice(meta_games_idx_available_pl1)
    meta_pi_pl1 = hp["meta_game_policy_distributions"][meta_pi_pl1_idx][
        POLICY_ID_PL1
    ]
    pl1_principal_i = hp["meta_game_matrix_pos_to_principal_idx"][
        meta_pi_pl1_idx
    ]

    n_meta_actions = 7
    payoff_matrix = copy.deepcopy(
        payoff_matrix_35x35[
            pl0_principal_i
            * n_meta_actions : (pl0_principal_i + 1)
            * n_meta_actions,
            pl1_principal_i
            * n_meta_actions : (pl1_principal_i + 1)
            * n_meta_actions,
            :,
        ]
    )
    print("cross_play_payoff_matrix", payoff_matrix)

    return payoff_matrix, meta_pi_pl1


def _select_cross_play_meta_policies_for_payoff_reading(
    hp, meta_games_idx_available
):
    meta_pi_pl0_idx = random.choice(meta_games_idx_available)
    meta_pi_pl0 = hp["meta_game_policy_distributions"][meta_pi_pl0_idx][
        POLICY_ID_PL0
    ]

    meta_games_idx_available_pl1 = copy.deepcopy(meta_games_idx_available)
    meta_games_idx_available_pl1.remove(meta_pi_pl0_idx)

    meta_pi_pl1_idx = random.choice(meta_games_idx_available_pl1)
    meta_pi_pl1 = hp["meta_game_policy_distributions"][meta_pi_pl1_idx][
        POLICY_ID_PL1
    ]

    return meta_pi_pl0, meta_pi_pl1, meta_pi_pl0_idx, meta_pi_pl1_idx


def _compute_joint_meta_pi_proba(meta_pi_pl0, meta_pi_pl1):
    meta_pi_pl0 = np.expand_dims(meta_pi_pl0, axis=-1)
    meta_pi_pl0 = np.expand_dims(meta_pi_pl0, axis=-1)
    meta_pi_pl0_ext = np.tile(meta_pi_pl0, (1, len(meta_pi_pl1), 2))
    meta_pi_pl1 = np.expand_dims(meta_pi_pl1, axis=0)
    meta_pi_pl1 = np.expand_dims(meta_pi_pl1, axis=-1)
    meta_pi_pl1_ext = np.tile(meta_pi_pl1, (len(meta_pi_pl0), 1, 2))
    joint_proba = meta_pi_pl0_ext * meta_pi_pl1_ext
    n_players = 2
    assert np.abs(joint_proba.sum() - n_players) < EPSILON, (
        f"joint_proba.sum()" f" {joint_proba.sum()}"
    )
    assert np.all(joint_proba >= 0.0 - EPSILON), f"joint_proba {joint_proba}"
    assert np.all(joint_proba <= 1.0 + EPSILON), f"joint_proba {joint_proba}"
    return joint_proba


def _evaluate_by_playing_in_base_game(hp):
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
    elif hp["base_game_policies"] == BASE_LOLA_EXACT:
        (
            stop_config,
            env_config,
            rllib_config,
            trainer,
            hp_eval,
        ) = _get_rllib_config_for_base_lola_exact_policy(hp)
    elif hp["base_game_policies"] == BASE_NEGOTIATION:
        (
            stop_config,
            env_config,
            rllib_config,
            trainer,
            hp_eval,
        ) = _get_rllib_config_for_base_negociation_policy(hp)
    else:
        raise ValueError()

    (
        rllib_config,
        stop_config,
    ) = _change_simple_rllib_config_for_final_base_game_eval(
        hp, rllib_config, stop_config
    )

    if hp["apply_announcement_protocol"]:
        rllib_config = _change_rllib_config_to_use_welfare_coordination(
            hp, rllib_config, meta_game_idx, env_config
        )
    else:
        rllib_config = _change_rllib_config_to_use_stochastic_populations(
            hp, rllib_config, meta_game_idx
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


def _get_rllib_config_for_base_lola_exact_policy(hp):
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

    return stop_config, env_config, rllib_config, trainer, lola_exact_hp


def _get_rllib_config_for_base_negociation_policy(hp):
    raise NotImplementedError()
    from marltoolbox.algos.alternating_offers import alt_offers_training

    lola_exact_hp = alt_offers_training.get_hyperparameters()
    (
        hp_eval,
        rllib_config,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    ) = alt_offers_training.generate_eval_config(lola_exact_hp)

    trainer = alt_offers_training.AltOffersTraining

    return stop_config, env_config, rllib_config, trainer, lola_exact_hp


def _change_simple_rllib_config_for_final_base_game_eval(
    hp, rllib_config, stop_config
):
    rllib_config["min_iter_time_s"] = 0.0
    rllib_config["timesteps_per_iteration"] = 0
    rllib_config["metrics_smoothing_episodes"] = 1
    if "max_steps" in rllib_config["env_config"].keys():
        rllib_config["rollout_fragment_length"] = rllib_config["env_config"][
            "max_steps"
        ]

    rllib_config["multiagent"]["policies_to_train"] = ["None"]
    rllib_config["callbacks"] = callbacks.merge_callbacks(
        callbacks.PolicyCallbacks,
        log.get_logging_callbacks_class(
            log_full_epi=True,
            # log_full_epi_interval=1,
            log_from_policy_in_evaluation=True,
        ),
    )
    rllib_config["seed"] = tune.sample_from(
        lambda spec: miscellaneous.get_random_seeds(1)[0]
    )
    if not hp["debug"]:
        stop_config["episodes_total"] = 100
    return rllib_config, stop_config


def _change_rllib_config_to_use_welfare_coordination(
    hp, rllib_config, meta_game_idx, env_config
):
    global payoffs_per_groups
    all_welfare_pairs_wt_payoffs = payoffs_per_groups[meta_game_idx][0]

    rllib_config["multiagent"]["policies_to_train"] = ["None"]
    policies = rllib_config["multiagent"]["policies"]
    for policy_idx, policy_id in enumerate(env_config["players_ids"]):
        policy_config_items = list(policies[policy_id])
        opp_policy_idx = (policy_idx + 1) % 2

        egalitarian_welfare_name = (
            "inequity aversion"
            if hp["base_game_policies"] == BASE_AMTFT
            else "egalitarian"
        )
        meta_policy_config = copy.deepcopy(welfare_coordination.DEFAULT_CONFIG)
        meta_policy_config.update(
            {
                "nested_policies": [
                    {
                        "Policy_class": copy.deepcopy(policy_config_items[0]),
                        "config_update": copy.deepcopy(policy_config_items[3]),
                    },
                ],
                "all_welfare_pairs_wt_payoffs": all_welfare_pairs_wt_payoffs,
                "solve_meta_game_after_init": False,
                "own_player_idx": policy_idx,
                "opp_player_idx": opp_policy_idx,
                "own_default_welfare_fn": egalitarian_welfare_name
                if policy_idx == 1
                else "utilitarian",
                "opp_default_welfare_fn": egalitarian_welfare_name
                if opp_policy_idx == 1
                else "utilitarian",
                "policy_id_to_load": policy_id,
                "policy_checkpoints": hp["base_ckpt_per_replicat"][
                    meta_game_idx
                ],
                "distrib_over_welfare_sets_to_annonce": hp[
                    "meta_game_policy_distributions"
                ][meta_game_idx][policy_id],
            }
        )
        policy_config_items[
            0
        ] = welfare_coordination.WelfareCoordinationTorchPolicy
        policy_config_items[3] = meta_policy_config
        policies[policy_id] = tuple(policy_config_items)

    return rllib_config


def _change_rllib_config_to_use_stochastic_populations(
    hp, rllib_config, meta_game_idx
):
    tmp_env = rllib_config["env"](rllib_config["env_config"])
    policies = rllib_config["multiagent"]["policies"]
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
    return rllib_config


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
    print('hp["actions_possible"]', hp["actions_possible"])
    for welfare_i in hp["actions_possible"]:
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
    player_1_payoffs = exp_analysis.extract_metrics_for_each_trials(
        tune_analysis, metric=hp_eval["x_axis_metric"]
    )
    player_2_payoffs = exp_analysis.extract_metrics_for_each_trials(
        tune_analysis, metric=hp_eval["y_axis_metric"]
    )
    print("player_1_payoffs", player_1_payoffs)
    print("player_2_payoffs", player_2_payoffs)
    return player_1_payoffs, player_2_payoffs


def _extract_coordination_metric(tune_analysis):
    coordination_success = exp_analysis.extract_metrics_for_each_trials(
        tune_analysis, metric="custom_metrics/coordination_success_mean"
    )
    coordination_success = [float(el) for el in coordination_success]

    # coordination_success = path.
    #     tune_analysis, metric="custom_metrics/coordination_success_mean"
    # )
    return coordination_success


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
    payoff_matrices = copy.deepcopy(hp["payoff_matrices"])

    policies = []
    for payoff_matrix in payoff_matrices:

        payoff_tables_per_player = [
            payoff_matrix[:, :, 0],
            payoff_matrix[:, :, 1],
        ]
        policy_player_1, policy_player_2 = _compute_policy_wt_alpha_rank(
            payoff_tables_per_player
        )

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


def _compute_policy_wt_alpha_rank(payoff_tables_per_player):
    from open_spiel.python.egt import alpharank
    from open_spiel.python.algorithms.psro_v2 import utils as psro_v2_utils

    joint_arank, alpha = alpharank.sweep_pi_vs_alpha(
        payoff_tables_per_player, return_alpha=True
    )
    print("alpha selected", alpha)
    (
        policy_player_1,
        policy_player_2,
    ) = psro_v2_utils.get_alpharank_marginals(
        payoff_tables_per_player, joint_arank
    )
    print("meta policy_player_1", policy_player_1)
    print("meta policy_player_2", policy_player_2)
    policy_player_1 = torch.tensor(policy_player_1)
    policy_player_2 = torch.tensor(policy_player_2)
    return policy_player_1, policy_player_2


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
        num_actions = payoff_matrix.shape[0]
        prd_initial_strategies = [
            np.random.dirichlet(np.ones(num_actions) * 1.5),
            np.random.dirichlet(np.ones(num_actions) * 1.5),
        ]
        if hp["meta_game_policies"] == META_REPLICATOR_DYNAMIC_ZERO_INIT:
            policy_player_1, policy_player_2 = projected_replicator_dynamics(
                payoff_tables_per_player,
                prd_gamma=0.0,
            )
        else:
            print("prd_initial_strategies", prd_initial_strategies)
            policy_player_1, policy_player_2 = projected_replicator_dynamics(
                payoff_tables_per_player,
                prd_gamma=0.0,
                prd_initial_strategies=prd_initial_strategies,
            )

        policy_player_1 = torch.tensor(policy_player_1)
        policy_player_2 = torch.tensor(policy_player_2)
        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
    print("replicator dynamic meta policies", policies)
    return policies


def _get_random_meta_policy(hp):
    payoff_matrices = copy.deepcopy(hp["payoff_matrices"])
    policies = []
    for payoff_matrix in payoff_matrices:
        num_actions_player_0 = payoff_matrix.shape[0]
        num_actions_player_1 = payoff_matrix.shape[1]

        policy_player_1 = (
            torch.ones(size=(num_actions_player_0,)) / num_actions_player_0
        )
        policy_player_2 = (
            torch.ones(size=(num_actions_player_1,)) / num_actions_player_1
        )
        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
    print("random meta policies", policies)
    return policies


def _train_meta_policy_using_robustness(hp):

    payoff_matrices = copy.deepcopy(hp["payoff_matrices"])
    policies = []
    for payoff_matrix in payoff_matrices:
        robustness_score_pl0 = payoff_matrix[:, :, 0].mean(axis=1)
        robustness_score_pl1 = payoff_matrix[:, :, 1].mean(axis=0)

        pl0_action = np.argmax(robustness_score_pl0, axis=0)
        pl1_action = np.argmax(robustness_score_pl1, axis=0)

        policy_player_1 = torch.zeros((payoff_matrix.shape[0],))
        policy_player_2 = torch.zeros((payoff_matrix.shape[1],))
        policy_player_1[pl0_action] = 1.0
        policy_player_2[pl1_action] = 1.0
        policies.append(
            {"player_row": policy_player_1, "player_col": policy_player_2}
        )
    print("replicator dynamic meta policies", policies)
    return policies


if __name__ == "__main__":
    debug_mode = True
    loop_over_main = True

    if loop_over_main:
        base_game_algo_to_eval = (
            BASE_LOLA_EXACT,
            # BASE_NEGOTIATION,
        )
        meta_game_algo_to_eval = (
            META_APLHA_RANK,
            META_APLHA_PURE,
            META_REPLICATOR_DYNAMIC,
            META_REPLICATOR_DYNAMIC_ZERO_INIT,
            META_RANDOM,
            META_PG,
            META_LOLA_EXACT,
            META_SOS,
            META_UNIFORM,
        )
        for base_game_algo in base_game_algo_to_eval:
            for meta_game_algo in meta_game_algo_to_eval:
                main(debug_mode, base_game_algo, meta_game_algo)
    else:
        main(debug_mode)
