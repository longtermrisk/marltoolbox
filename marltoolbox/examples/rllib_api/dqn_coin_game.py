import os

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS

from marltoolbox.algos import augmented_dqn
from marltoolbox.envs import coin_game
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import log, miscellaneous, exploration


def main(debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("DQN_CG")

    hparams = get_hyperparameters(seeds, debug, exp_name)

    rllib_config, stop_config = get_rllib_configs(hparams)

    tune_analysis = train_dqn_and_plot_logs(hparams, rllib_config, stop_config)

    return tune_analysis


def get_hyperparameters(seeds, debug, exp_name):
    """Get hyperparameters for the Coin Game env and DQN agents"""

    hparams = {
        "seeds": seeds,
        "debug": debug,
        "exp_name": exp_name,
        "n_steps_per_epi": 100,
        "n_epi": 4000,
        "buf_frac": 0.5,
        "last_exploration_temp_value": 0.003,
        "bs_epi_mul": 4,
        "plot_keys": coin_game.PLOT_KEYS
        + aggregate_and_plot_tensorboard_data.PLOT_KEYS,
        "plot_assemblage_tags": coin_game.PLOT_ASSEMBLAGE_TAGS
        + aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS,
    }
    return hparams


def get_rllib_configs(hp, env_class=None):
    stop_config = {
        "episodes_total": 2 if hp["debug"] else hp["n_epi"],
    }

    env_config = {
        "players_ids": ["player_red", "player_blue"],
        "max_steps": hp["n_steps_per_epi"],
        "grid_size": 3,
        "get_additional_info": True,
        "buf_frac": hp["buf_frac"],
        "bs_epi_mul": hp["bs_epi_mul"],
    }

    env_class = coin_game.CoinGame if env_class is None else env_class
    rllib_config = {
        "env": env_class,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    augmented_dqn.MyDQNTorchPolicy,
                    env_class(env_config).OBSERVATION_SPACE,
                    env_class.ACTION_SPACE,
                    {},
                ),
                env_config["players_ids"][1]: (
                    augmented_dqn.MyDQNTorchPolicy,
                    env_class(env_config).OBSERVATION_SPACE,
                    env_class.ACTION_SPACE,
                    {},
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        # === DQN Models ===
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": tune.sample_from(
            lambda spec: int(spec.config["env_config"]["max_steps"] * 30)
        ),
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": tune.sample_from(
            lambda spec: int(
                spec.config["env_config"]["max_steps"]
                * spec.stop["episodes_total"]
                * spec.config["env_config"]["buf_frac"]
            )
        ),
        # Whether to use dueling dqn
        "dueling": True,
        # Whether to use double dqn
        "double_q": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        "rollout_fragment_length": tune.sample_from(
            lambda spec: spec.config["env_config"]["max_steps"]
        ),
        "training_intensity": tune.sample_from(
            lambda spec: spec.config["num_envs_per_worker"]
            * max(spec.config["num_workers"], 1)
            * 40
        ),
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": tune.sample_from(
            lambda spec: int(
                spec.config["env_config"]["max_steps"]
                * spec.config["env_config"]["bs_epi_mul"]
            )
        ),
        "batch_mode": "complete_episodes",
        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
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
                        (0, 1.0),
                        (
                            int(
                                spec.config["env_config"]["max_steps"]
                                * spec.stop["episodes_total"]
                                * 0.20
                            ),
                            0.45,
                        ),
                        (
                            int(
                                spec.config["env_config"]["max_steps"]
                                * spec.stop["episodes_total"]
                                * 0.9
                            ),
                            hp["last_exploration_temp_value"],
                        ),
                    ],
                    outside_value=hp["last_exploration_temp_value"],
                    framework="torch",
                )
            ),
        },
        # Size of batches collected from each worker.
        "model": {
            "dim": env_config["grid_size"],
            # [Channel, [Kernel, Kernel], Stride]]
            "conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1]],
            "fcnet_hiddens": [64, 64],
        },
        "hiddens": [32],
        "gamma": 0.96,
        "optimizer": {
            "sgd_momentum": 0.9,
        },
        "lr": 0.1,
        "lr_schedule": tune.sample_from(
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
                    ),
                    spec.config.lr / 1e9,
                ),
            ]
        ),
        "seed": tune.grid_search(hp["seeds"]),
        "callbacks": log.get_logging_callbacks_class(),
        "framework": "torch",
        "logger_config": {
            "wandb": {
                "project": "DQN_CG",
                "group": hp["exp_name"],
                "api_key_file": os.path.join(
                    os.path.dirname(__file__), "../../../api_key_wandb"
                ),
                "log_config": True,
            },
        },
        "num_envs_per_worker": 16,
        "num_workers": 0,
        # "log_level": "INFO",
    }

    return rllib_config, stop_config


def train_dqn_and_plot_logs(hp, rllib_config, stop_config):
    ray.init(num_cpus=os.cpu_count(), local_mode=hp["debug"])
    tune_analysis = tune.run(
        DQNTrainer,
        config=rllib_config,
        stop=stop_config,
        name=hp["exp_name"],
        log_to_file=not hp["debug"],
        loggers=None if hp["debug"] else DEFAULT_LOGGERS + (WandbLogger,),
    )
    if not hp["debug"]:
        aggregate_and_plot_tensorboard_data.add_summary_plots(
            main_path=os.path.join("~/ray_results/", hp["exp_name"]),
            plot_keys=hp["plot_keys"],
            plot_assemble_tags_in_one_plot=hp["plot_assemblage_tags"],
        )
    ray.shutdown()
    return tune_analysis


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
