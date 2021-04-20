from ray import tune

from marltoolbox.envs import coin_game, ssd_mixed_motive_coin_game
from marltoolbox.examples.rllib_api import dqn_coin_game
from marltoolbox.examples.rllib_api.dqn_wt_welfare import (
    modify_dqn_rllib_config_to_use_welfare,
)
from marltoolbox.utils import log, miscellaneous, postprocessing


def main(debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("DQN_CG_speed_search")

    env = "CoinGame"
    # env = "SSDMixedMotiveCoinGame"
    # welfare_to_use = None
    # welfare_to_use = postprocessing.WELFARE_UTILITARIAN
    welfare_to_use = postprocessing.WELFARE_INEQUITY_AVERSION

    if "SSDMixedMotiveCoinGame" in env:
        env_class = ssd_mixed_motive_coin_game.SSDMixedMotiveCoinGame
    else:
        env_class = coin_game.CoinGame

    hparams = dqn_coin_game.get_hyperparameters(seeds, debug, exp_name)

    rllib_config, stop_config = dqn_coin_game.get_rllib_configs(
        hparams, env_class=env_class
    )

    if welfare_to_use is not None:
        rllib_config = modify_dqn_rllib_config_to_use_welfare(
            rllib_config, welfare_to_use
        )

    rllib_config, stop_config = _add_search_to_config(
        rllib_config, stop_config, hparams
    )
    tune_analysis = dqn_coin_game.train_dqn_and_plot_logs(
        hparams, rllib_config, stop_config
    )

    return tune_analysis


def _add_search_to_config(rllib_config, stop_config, hp):
    # rllib_config["num_gpus"] = 0.33
    # rllib_config["num_gpus_per_worker"] = 0.33
    rllib_config["num_envs_per_worker"] = tune.grid_search([128, 64, 32, 16])
    rllib_config["lr"] = 0.2 if hp["debug"] else 0.2
    # rllib_config["env_config"]["max_steps"] = 10 if hp["debug"] else 100
    rllib_config["num_workers"] = (
        0 if hp["debug"] else tune.grid_search([0, 3, 9])
    )
    stop_config["episodes_total"] = (
        10 if hp["debug"] else tune.grid_search([2000, 4000])
    )
    # rllib_config["model"] = {
    #     "dim": 3,
    #     "conv_filters": [[16, [3, 3], 1], [16, [3, 3], 1]],
    #     "fcnet_hiddens": [64, 64],
    # }
    # rllib_config["hiddens"] = [32]
    # rllib_config["env_config"].update(
    #     {
    #         "max_steps": 100,
    #         "grid_size": 3,
    #         "temp_mid_step": 0.6,
    #         "bs_epi_mul": 4 if hp["debug"] else 4,
    #     }
    # )
    # rllib_config["training_intensity"] = tune.sample_from(
    #     lambda spec: spec.config["num_envs_per_worker"]
    #     * max(1, spec.config["num_workers"])
    #     * 10
    # )

    # rllib_config["exploration_config"] = {
    #     # The Exploration class to use. In the simplest case,
    #     # this is the name (str) of any class present in the
    #     # `rllib.utils.exploration` package.
    #     # You can also provide the python class directly or
    #     # the full location of your class (e.g.
    #     # "ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy").
    #     # "type": exploration.SoftQSchedule,
    #     "type": exploration.SoftQSchedule,
    #     # Add constructor kwargs here (if any).
    #     "temperature_schedule": tune.sample_from(
    #         lambda spec: PiecewiseSchedule(
    #             endpoints=[
    #                 (0, 2.0),
    #                 (
    #                     int(
    #                         spec.config["env_config"]["max_steps"]
    #                         * spec.stop["episodes_total"]
    #                         * 0.20
    #                     ),
    #                     0.5,
    #                 ),
    #                 (
    #                     int(
    #                         spec.config["env_config"]["max_steps"]
    #                         * spec.stop["episodes_total"]
    #                         * spec.config["env_config"]["temp_mid_step"]
    #                     ),
    #                     hp["last_exploration_temp_value"],
    #                 ),
    #             ],
    #             outside_value=hp["last_exploration_temp_value"],
    #             framework="torch",
    #         )
    #     ),
    # }
    # rllib_config["train_batch_size"] = tune.sample_from(
    #     lambda spec: int(
    #         spec.config["env_config"]["max_steps"]
    #         * spec.config["env_config"]["bs_epi_mul"]
    #     )
    # )

    return rllib_config, stop_config


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
