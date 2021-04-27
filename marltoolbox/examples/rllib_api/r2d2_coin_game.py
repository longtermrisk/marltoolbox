import os

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.rllib.agents.dqn.r2d2_torch_policy import postprocess_nstep_and_prio
from ray.rllib.utils.schedules import PiecewiseSchedule

from marltoolbox.examples.rllib_api import dqn_coin_game, dqn_wt_welfare
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import log, miscellaneous, postprocessing
from marltoolbox.algos import augmented_r2d2
from marltoolbox.envs import coin_game, ssd_mixed_motive_coin_game


def main(debug):
    """Train R2D2 agent in the Coin Game environment"""

    # env = "CoinGame"
    env = "SSDMixedMotiveCoinGame"
    # welfare_to_use = None
    welfare_to_use = postprocessing.WELFARE_UTILITARIAN
    # welfare_to_use = postprocessing.WELFARE_INEQUITY_AVERSION

    rllib_config, stop_config, hparams = _get_config_and_hp_for_training(
        debug, env, welfare_to_use
    )

    tune_analysis = _train_dqn(hparams, rllib_config, stop_config)

    _plot_log_aggregates(hparams)

    return tune_analysis


def _get_config_and_hp_for_training(debug, env, welfare_to_use):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("R2D2_CG")
    if "SSDMixedMotiveCoinGame" in env:
        env_class = ssd_mixed_motive_coin_game.SSDMixedMotiveCoinGame
    else:
        env_class = coin_game.CoinGame

    hparams = dqn_coin_game.get_hyperparameters(seeds, debug, exp_name)
    rllib_config, stop_config = dqn_coin_game.get_rllib_configs(
        hparams, env_class=env_class
    )
    rllib_config, stop_config = _adapt_configs_for_r2d2(
        rllib_config, stop_config, hparams
    )
    if welfare_to_use is not None:
        rllib_config = modify_r2d2_rllib_config_to_use_welfare(
            rllib_config, welfare_to_use
        )
    return rllib_config, stop_config, hparams


def modify_r2d2_rllib_config_to_use_welfare(rllib_config, welfare_to_use):
    r2d2_torch_policy_class_wt_welfare = (
        _get_r2d2_policy_class_wt_welfare_preprocessing()
    )
    rllib_config = dqn_wt_welfare.modify_rllib_config_to_use_welfare(
        rllib_config,
        welfare_to_use,
        policy_class_wt_welfare=r2d2_torch_policy_class_wt_welfare,
    )
    return rllib_config


def _get_r2d2_policy_class_wt_welfare_preprocessing():
    r2d2_torch_policy_class_wt_welfare = (
        augmented_r2d2.MyR2D2TorchPolicy.with_updates(
            postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
                postprocessing.welfares_postprocessing_fn(),
                postprocess_nstep_and_prio,
            )
        )
    )
    return r2d2_torch_policy_class_wt_welfare


def _adapt_configs_for_r2d2(rllib_config, stop_config, hp):
    rllib_config["logger_config"]["wandb"]["project"] = "R2D2_CG"
    rllib_config["model"]["use_lstm"] = True
    rllib_config["burn_in"] = 0
    rllib_config["zero_init_states"] = False
    rllib_config["use_h_function"] = False

    rllib_config = _replace_class_of_policies_by(
        augmented_r2d2.MyR2D2TorchPolicy,
        rllib_config,
    )

    if not hp["debug"]:
        # rllib_config["env_config"]["training_intensity"] = 40
        # rllib_config["lr"] = 0.1
        # rllib_config["training_intensity"] = tune.sample_from(
        #     lambda spec: spec.config["num_envs_per_worker"]
        #     * spec.config["env_config"]["training_intensity"]
        #     * max(1, spec.config["num_workers"])
        # )
        stop_config["episodes_total"] = 8000
        rllib_config["model"]["lstm_cell_size"] = 16
        rllib_config["model"]["max_seq_len"] = 20
        # rllib_config["env_config"]["buf_frac"] = 0.5
        # rllib_config["hiddens"] = [32]
        # rllib_config["model"]["fcnet_hiddens"] = [64, 64]
        # rllib_config["model"]["conv_filters"] = tune.grid_search(
        #     [
        #         [[32, [3, 3], 1], [32, [3, 3], 1]],
        #         [[64, [3, 3], 1], [64, [3, 3], 1]],
        #     ]
        # )

        rllib_config["env_config"]["temp_ratio"] = (
            1.0 if hp["debug"] else tune.grid_search([0.75, 1.0])
        )
        rllib_config["env_config"]["interm_temp_ratio"] = (
            1.0 if hp["debug"] else tune.grid_search([0.75, 1.0])
        )
        rllib_config["env_config"]["last_exploration_t"] = (
            0.6 if hp["debug"] else tune.grid_search([0.9, 0.99])
        )
        rllib_config["env_config"]["last_exploration_temp_value"] = (
            1.0 if hp["debug"] else 0.003
        )
        rllib_config["env_config"]["interm_exploration_t"] = (
            0.2 if hp["debug"] else tune.grid_search([0.2, 0.6])
        )
        rllib_config["exploration_config"][
            "temperature_schedule"
        ] = tune.sample_from(
            lambda spec: PiecewiseSchedule(
                endpoints=[
                    (0, 1.0 * spec.config["env_config"]["temp_ratio"]),
                    (
                        int(
                            spec.config["env_config"]["max_steps"]
                            * spec.stop["episodes_total"]
                            * spec.config["env_config"]["interm_exploration_t"]
                        ),
                        0.6
                        * spec.config["env_config"]["temp_ratio"]
                        * spec.config["env_config"]["interm_temp_ratio"],
                    ),
                    (
                        int(
                            spec.config["env_config"]["max_steps"]
                            * spec.stop["episodes_total"]
                            * spec.config["env_config"]["last_exploration_t"]
                        ),
                        spec.config["env_config"][
                            "last_exploration_temp_value"
                        ],
                    ),
                ],
                outside_value=spec.config["env_config"][
                    "last_exploration_temp_value"
                ],
                framework="torch",
            )
        )

        # rllib_config["env_config"]["bs_epi_mul"] = (
        #     4 if hp["debug"] else tune.grid_search([4, 8, 16])
        # )
        rllib_config["env_config"]["interm_lr_ratio"] = (
            0.5
            if hp["debug"]
            else tune.grid_search([0.5 * 3, 0.5, 0.5 / 3, 0.5 / 9])
        )
        rllib_config["env_config"]["interm_lr_t"] = (
            0.5 if hp["debug"] else tune.grid_search([0.25, 0.5, 0.75])
        )
        rllib_config["lr"] = 0.1 if hp["debug"] else 0.1
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
                        * spec.config["env_config"]["interm_lr_t"]
                    ),
                    spec.config.lr
                    * spec.config["env_config"]["interm_lr_ratio"],
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
    else:
        rllib_config["model"]["max_seq_len"] = 2
        rllib_config["model"]["lstm_cell_size"] = 8

    return rllib_config, stop_config


def _replace_class_of_policies_by(new_policy_class, rllib_config):
    policies = rllib_config["multiagent"]["policies"]
    for policy_id in policies.keys():
        policy = list(policies[policy_id])
        policy[0] = new_policy_class
        policies[policy_id] = tuple(policy)
    return rllib_config


def _train_dqn(hp, rllib_config, stop_config):
    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=hp["debug"])
    tune_analysis = tune.run(
        dqn.r2d2.R2D2Trainer,
        config=rllib_config,
        stop=stop_config,
        name=hp["exp_name"],
        log_to_file=not hp["debug"],
        loggers=None if hp["debug"] else DEFAULT_LOGGERS + (WandbLogger,),
    )
    ray.shutdown()
    return tune_analysis


def _plot_log_aggregates(hp):
    if not hp["debug"]:
        aggregate_and_plot_tensorboard_data.add_summary_plots(
            main_path=os.path.join("~/ray_results/", hp["exp_name"]),
            plot_keys=hp["plot_keys"],
            plot_assemble_tags_in_one_plot=hp["plot_assemblage_tags"],
        )


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
