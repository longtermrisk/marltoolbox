import os

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn import r2d2
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS

from marltoolbox.examples.rllib_api import dqn_coin_game
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import log, miscellaneous
from marltoolbox.algos import augmented_r2d2


def main(debug):
    """Train R2D2 agent in the Coin Game environment"""

    rllib_config, stop_config, hparams = _get_config_and_hp_for_training(debug)

    tune_analysis = _train_dqn(hparams, rllib_config, stop_config)

    _plot_log_aggregates(hparams)

    return tune_analysis


def _get_config_and_hp_for_training(debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("R2D2_CG")

    hparams = dqn_coin_game.get_hyperparameters(seeds, debug, exp_name)
    rllib_config, stop_config = dqn_coin_game.get_rllib_configs(hparams)
    rllib_config, stop_config = _adapt_configs_for_r2d2(
        rllib_config, stop_config
    )
    return rllib_config, stop_config, hparams


def _adapt_configs_for_r2d2(rllib_config, stop_config):
    rllib_config["logger_config"]["wandb"]["project"] = "R2D2_CG"
    rllib_config["model"]["use_lstm"] = True
    # rllib_config["num_workers"] = 0
    rllib_config = _replace_class_of_policies_by(
        augmented_r2d2.MyR2D2TorchPolicy, rllib_config
    )
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
        dqn.R2D2Trainer,
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
