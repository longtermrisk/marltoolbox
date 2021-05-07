from ray import tune
from ray.rllib.utils.schedules import PiecewiseSchedule

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
    env = "SSDMixedMotiveCoinGame"
    welfare_to_use = None
    welfare_to_use = postprocessing.WELFARE_UTILITARIAN
    # welfare_to_use = postprocessing.WELFARE_INEQUITY_AVERSION

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
    assert hp["last_exploration_temp_value"] == 0.01

    stop_config["episodes_total"] = 10 if hp["debug"] else 8000
    # rllib_config["lr"] = 0.1
    # rllib_config["env_config"]["training_intensity"] = (
    #     20 if hp["debug"] else 40
    # )
    # rllib_config["training_intensity"] = tune.sample_from(
    #     lambda spec: spec.config["num_envs_per_worker"]
    #     * max(spec.config["num_workers"], 1)
    #     * spec.config["env_config"]["training_intensity"]
    # )
    # rllib_config["env_config"]["temp_ratio"] = (
    #     1.0 if hp["debug"] else tune.grid_search([1.0, 0.5, 2.0])
    # )
    # rllib_config["env_config"]["interm_temp_ratio"] = (
    #     1.0 if hp["debug"] else tune.grid_search([1.0, 5.0, 2.0, 3.0, 10.0])
    # )
    # rllib_config["env_config"]["last_exploration_t"] = (
    #     0.6 if hp["debug"] else 0.9
    # )
    # rllib_config["env_config"]["last_exploration_temp_value"] = (
    #     1.0 if hp["debug"] else 0.003
    # )
    # rllib_config["exploration_config"][
    #     "temperature_schedule"
    # ] = tune.sample_from(
    #     lambda spec: PiecewiseSchedule(
    #         endpoints=[
    #             (0, 0.5 * spec.config["env_config"]["temp_ratio"]),
    #             (
    #                 int(
    #                     spec.config["env_config"]["max_steps"]
    #                     * spec.stop["episodes_total"]
    #                     * 0.20
    #                 ),
    #                 0.1
    #                 * spec.config["env_config"]["temp_ratio"]
    #                 * spec.config["env_config"]["interm_temp_ratio"],
    #             ),
    #             (
    #                 int(
    #                     spec.config["env_config"]["max_steps"]
    #                     * spec.stop["episodes_total"]
    #                     * spec.config["env_config"]["last_exploration_t"]
    #                 ),
    #                 spec.config["env_config"]["last_exploration_temp_value"],
    #             ),
    #         ],
    #         outside_value=spec.config["env_config"][
    #             "last_exploration_temp_value"
    #         ],
    #         framework="torch",
    #     )
    # )

    rllib_config["env_config"]["bs_epi_mul"] = (
        4 if hp["debug"] else tune.grid_search([4, 8, 16])
    )
    rllib_config["env_config"]["interm_lr_ratio"] = (
        0.5 if hp["debug"] else tune.grid_search([0.5 * 3, 0.5, 0.5 / 3])
    )
    rllib_config["lr"] = (
        0.1 if hp["debug"] else tune.grid_search([0.1, 0.2, 0.4])
    )
    rllib_config["lr_schedule"] = tune.sample_from(
        lambda spec: [
            (0, 0.0),
            (
                int(
                    spec.config["env_config"]["max_steps"]
                    * spec.stop["episodes_total"]
                    * 0.05
                ),
                spec.config.lr,
            ),
            (
                int(
                    spec.config["env_config"]["max_steps"]
                    * spec.stop["episodes_total"]
                    * 0.5
                ),
                spec.config.lr * spec.config["env_config"]["interm_lr_ratio"],
            ),
            (
                int(
                    spec.config["env_config"]["max_steps"]
                    * spec.stop["episodes_total"]
                ),
                spec.config.lr / 1e9,
            ),
        ]
    )

    return rllib_config, stop_config


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
