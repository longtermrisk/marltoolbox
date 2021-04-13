import json
import logging

from marltoolbox.algos import welfare_coordination
from marltoolbox.experiments.rllib_api import amtft_various_env

logger = logging.getLogger(__name__)


def main(debug):
    hparams = _get_hyperparameters(debug)

    meta_policy_config, all_welfare_pairs_wt_payoffs, all_trained_policies = \
        _get_meta_policy_config(hparams)

    player_meta_policy = welfare_coordination.WelfareCoordinationTorchPolicy(
        hparams["env_class"].OBSERVATION_SPACE,
        hparams["env_class"].ACTION_SPACE,
        meta_policy_config
    )
    player_meta_policy.setup_meta_game(
        all_welfare_pairs_wt_payoffs,
        all_trained_policies)

    for tau_x_10 in range(0, 11, 1):
        player_meta_policy.solve_meta_game(tau_x_10 / 10)


def _get_hyperparameters(debug):
    env = "CoinGame"
    train_n_replicates = 1
    filter_utilitarian = False

    hp = amtft_various_env.get_hyperparameters(
        debug, train_n_replicates, filter_utilitarian, env)

    hp.update({
        "json_file": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-2/"
                     "amTFT/2021_04_09/08_06_53/eval/2021_04_09/20_01_53/"
                     "self_and_cross_play_policy_reward_mean_player_blue_vs_"
                     "policy_reward_mean_player_red_matrix.json",
        "player1": "player_red",
        "player2": "player_blue",
    })

    return hp


def _get_meta_policy_config(hp):
    all_welfare_pairs_wt_payoffs = _get_all_welfare_pairs_wt_payoffs(hp)

    return {}, all_welfare_pairs_wt_payoffs, []


def _get_all_welfare_pairs_wt_payoffs(hp):
    with open(hp["json_file"]) as json_file:
        json_data = json.load(json_file)
    print("json_data", json_data)

    cross_play_data = _keep_only_cross_play_values(json_data)
    cross_play_means = _keep_only_mean_values(cross_play_data)
    all_welfare_pairs_wt_payoffs = _order_players(cross_play_means, hp)

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
    return welfare_coordination.WelfareCoordinationTorchPolicy. \
        _from_pair_of_welfare_names_to_key(*both_welfare_fn)


def _keep_only_mean_values(cross_play_data):
    return {
        wekfare_pair_k:
            {player_k: player_v["mean"]
             for player_k, player_v in eval_dict.items()}
        for wekfare_pair_k, eval_dict in cross_play_data.items()
    }


def _order_players(cross_play_means, hp):
    return {
        wekfare_pair_k:
            [
                eval_dict[
                    [k for k in eval_dict.keys() if hp["player1"] in k][0]],
                eval_dict[
                    [k for k in eval_dict.keys() if hp["player2"] in k][0]],
            ]
        for wekfare_pair_k, eval_dict in cross_play_means.items()
    }


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
