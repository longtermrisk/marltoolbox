import os

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.utils.schedules import PiecewiseSchedule

from marltoolbox.algos import augmented_dqn
from marltoolbox.envs import mixed_motive_coin_game
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import log, miscellaneous, exploration


def main(debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("DQN_MMCG")

    hparams = {
        "n_steps_per_epi": 100,
        "n_epi": 4000,
        "buf_frac": 0.3,
        "last_exploration_temp_value":0.1,
        "plot_keys":
            mixed_motive_coin_game.PLOT_KEYS +
            aggregate_and_plot_tensorboard_data.PLOT_KEYS,
        "plot_assemblage_tags":
            mixed_motive_coin_game.PLOT_ASSEMBLAGE_TAGS +
            aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS,
    }
    hparams["temperature_schedule"] = PiecewiseSchedule(
        endpoints=[
            (0, 2.0),
            (int(hparams["n_steps_per_epi"] * hparams["n_epi"] *
                 0.20), 0.5),
            (int(hparams["n_steps_per_epi"] * hparams["n_epi"] *
                 0.60),
             hparams["last_exploration_temp_value"])],
        outside_value=hparams["last_exploration_temp_value"],
        framework="torch")

    stop = {
        "training_iteration": 2 if debug else hparams["n_epi"],
    }

    env_config = {
        "players_ids": ["player_red", "player_blue"],
        "max_steps": hparams["n_steps_per_epi"],
        "grid_size": 3,
        "get_additional_info": True,
    }

    env_class = mixed_motive_coin_game.MixedMotiveCoinGame
    rllib_config = {
        "env": env_class,
        "env_config": env_config,

        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    augmented_dqn.MyDQNTorchPolicy,
                    env_class(env_config).OBSERVATION_SPACE,
                    env_class.ACTION_SPACE,
                    {}),
                env_config["players_ids"][1]: (
                    augmented_dqn.MyDQNTorchPolicy,
                    env_class(env_config).OBSERVATION_SPACE,
                    env_class.ACTION_SPACE,
                    {}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        # === DQN Models ===

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 3 * hparams["n_steps_per_epi"],
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(hparams["n_steps_per_epi"] * hparams["n_epi"] *
                           hparams["buf_frac"]),
        # Whether to use dueling dqn
        "dueling": False,
        # Whether to use double dqn
        "double_q": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,

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
            "temperature_schedule": hparams["temperature_schedule"],
        },

        # Size of batches collected from each worker.
        "model": {
            "dim": env_config["grid_size"],
            # [Channel, [Kernel, Kernel], Stride]]
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]]
        },
        "gamma": 0.5,
        "optimizer": {"sgd_momentum": 0.9,},
        "lr": 0.4,
        "lr_schedule": tune.sample_from(lambda spec: [
            (0,
             0.0),
            (int(hparams["n_steps_per_epi"] * hparams["n_epi"] * 0.05),
             spec.config.lr),
            (int(hparams["n_steps_per_epi"] * hparams["n_epi"]),
             spec.config.lr / 1e9)
        ]),

        "seed": tune.grid_search(seeds),
        "callbacks": log.get_logging_callbacks_class(),
        "framework": "torch",
    }

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)
    tune_analysis = tune.run(DQNTrainer,
                             config=rllib_config,
                             stop=stop,
                             name=exp_name)
    if not debug:
        aggregate_and_plot_tensorboard_data.add_summary_plots(
            main_path=os.path.join("~/ray_results/", exp_name),
            plot_keys=hparams["plot_keys"],
            plot_assemble_tags_in_one_plot=hparams["plot_assemblage_tags"],
        )
    ray.shutdown()
    return tune_analysis


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
