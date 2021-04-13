from ray import tune
from ray.rllib.utils.schedules import PiecewiseSchedule

from marltoolbox.envs import coin_game, ssd_mixed_motive_coin_game
from marltoolbox.examples.rllib_api.dqn_coin_game import (
    _get_hyperparameters,
    _get_rllib_configs,
    _train_dqn_and_plot_logs,
)
from marltoolbox.examples.rllib_api.dqn_wt_welfare import _modify_policy_to_use_welfare
from marltoolbox.utils import log, miscellaneous, postprocessing, exploration


def main(debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("DQN_CG_speed_search")

    env = "CoinGame"
    # env = "SSDMixedMotiveCoinGame"
    welfare_to_use = None
    # welfare_to_use = postprocessing.WELFARE_UTILITARIAN
    welfare_to_use = postprocessing.WELFARE_INEQUITY_AVERSION

    if "SSDMixedMotiveCoinGame" in env:
        env_class = ssd_mixed_motive_coin_game.SSDMixedMotiveCoinGame
    else:
        env_class = coin_game.CoinGame

    hparams = _get_hyperparameters(seeds, debug, exp_name)

    rllib_config, stop_config = _get_rllib_configs(hparams, env_class=env_class)

    if welfare_to_use is not None:
        rllib_config = _modify_policy_to_use_welfare(rllib_config, welfare_to_use)

    rllib_config, stop_config = _add_search_to_config(
        rllib_config, stop_config, hparams
    )
    tune_analysis = _train_dqn_and_plot_logs(hparams, rllib_config, stop_config)

    return tune_analysis


def _add_search_to_config(rllib_config, stop_config, hp):
    rllib_config["lr"] = 0.1
    rllib_config["model"] = {
        "dim": 3,
        "conv_filters": tune.grid_search(
            [
                [[16, [3, 3], 1], [16, [3, 3], 1]],
                [[8, [3, 3], 1], [8, [3, 3], 1]],
                [[4, [3, 3], 1], [4, [3, 3], 1]],
            ]
        ),
        "fcnet_hiddens": tune.grid_search(
            [[8, 8], [16, 16], [32, 32], [64, 64], [128, 128], [256, 256]]
        ),
    }
    rllib_config["hiddens"] = tune.grid_search([[4], [8], [16], [32]])
    rllib_config["env_config"] = {
        "players_ids": ["player_red", "player_blue"],
        "max_steps": 100,
        "grid_size": 3,
        "get_additional_info": True,
        "temp_mid_step": 0.6,
        "bs_epi_mul": 4,
    }
    rllib_config["training_intensity"] = 10

    stop_config["episodes_total"] = 2000

    rllib_config["exploration_config"] = {
        # The Exploration class to use. In the simplest case,
        # this is the name (str) of any class present in the
        # `rllib.utils.exploration` package.
        # You can also provide the python class directly or
        # the full location of your class (e.g.
        # "ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy").
        # "type": exploration.SoftQSchedule,
        "type": exploration.SoftQSchedule,
        # Add constructor kwargs here (if any).
        "temperature_schedule": tune.sample_from(
            lambda spec: PiecewiseSchedule(
                endpoints=[
                    (0, 2.0),
                    (
                        int(
                            spec.config["env_config"]["max_steps"]
                            * spec.stop["episodes_total"]
                            * 0.20
                        ),
                        0.5,
                    ),
                    (
                        int(
                            spec.config["env_config"]["max_steps"]
                            * spec.stop["episodes_total"]
                            * spec.config["env_config"]["temp_mid_step"]
                        ),
                        hp["last_exploration_temp_value"],
                    ),
                ],
                outside_value=hp["last_exploration_temp_value"],
                framework="torch",
            )
        ),
    }
    rllib_config["train_batch_size"] = tune.sample_from(
        lambda spec: int(
            spec.config["env_config"]["max_steps"]
            * spec.config["env_config"]["bs_epi_mul"]
        )
    )

    return rllib_config, stop_config


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
