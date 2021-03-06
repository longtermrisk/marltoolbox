import copy
import os

import ray
from ray import tune
from ray.rllib.agents.dqn.dqn_torch_policy import postprocess_nstep_and_prio
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS

from marltoolbox.algos import ltft, augmented_dqn
from marltoolbox.algos.exploiters.influence_evader import (
    InfluenceEvaderTorchPolicy,
)
from marltoolbox.envs import (
    matrix_sequential_social_dilemma,
    vectorized_coin_game,
    mixed_motive_coin_game,
)
from marltoolbox.envs.utils.wrappers import (
    add_RewardUncertaintyEnvClassWrapper,
)
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import (
    log,
    miscellaneous,
    exploration,
    postprocessing,
    callbacks,
)


def main(debug, env=None, train_n_replicates=None, against_naive_opp=False):
    """Train 2 agents in self play and then against an exploiter"""
    hparameters, exp_name = _get_hyparameters(
        debug, env, train_n_replicates, against_naive_opp
    )
    rllib_config, env_config, stop = _get_rllib_config(hparameters)

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)

    # tune_analysis_self_play = None
    tune_analysis_self_play = _train_in_self_play(
        rllib_config, stop, exp_name, hparameters
    )

    # tune_analysis_naive_opponent = None
    tune_analysis_against_opponent = _train_against_opponent(
        hparameters, rllib_config, stop, exp_name, env_config
    )

    ray.shutdown()
    return tune_analysis_self_play, tune_analysis_against_opponent


def _get_hyparameters(
    debug, env=None, train_n_replicates=None, against_naive_opp=False
):
    if debug:
        train_n_replicates = 1
    elif train_n_replicates is None:
        train_n_replicates = 4

    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("LTFT")

    hparameters = {
        "seeds": seeds,
        "debug": debug,
        "exp_name": exp_name,
        "hiddens": [64],
        "log_n_points": 260,
        "clustering_distance": 0.2,
        "gamma": 0.96,
        "env_name": "IteratedPrisonersDilemma" if env is None else env,
        # "env_name": "CoinGame" if env is None else env,
        "reward_uncertainty_std": 0.1,
        # "against_evader_exploiter": None,
        "against_evader_exploiter": {
            "start_exploit": 0.75,
            "copy_weights_delay": 0.05,
        }
        if not against_naive_opp
        else None,
    }

    hparameters = _modify_hyperparams_for_the_selected_env(hparameters)

    return hparameters, exp_name


def _modify_hyperparams_for_the_selected_env(hp):
    hp["plot_keys"] = (
        ltft.PLOT_KEYS + aggregate_and_plot_tensorboard_data.PLOT_KEYS
    )
    hp["plot_assemblage_tags"] = (
        ltft.PLOT_ASSEMBLAGE_TAGS
        + aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
    )
    hp["training_intensity"] = None

    if "CoinGame" in hp["env_name"]:
        hp["sgd_momentum"] = 0.9
        hp["both_players_can_pick_the_same_coin"] = True
        hp["buf_frac"] = 0.125
        hp["n_steps_per_epi"] = 100
        hp["spl_lr_mul"] = 3
        hp["bs_epi_mul"] = 4
        hp["bs_epi_mul_spl"] = 1
        hp["base_lr"] = 0.4
        hp["n_epi"] = 10 if hp["debug"] else 4000
        hp["training_intensity"] = 4
        hp["last_exploration_temp_value"] = 0.01

        hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.20), 0.5),
                (
                    int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.60),
                    hp["last_exploration_temp_value"],
                ),
            ],
            outside_value=hp["last_exploration_temp_value"],
            framework="torch",
        )

        hp["plot_keys"] += vectorized_coin_game.PLOT_KEYS
        hp["plot_assemblage_tags"] += vectorized_coin_game.PLOT_ASSEMBLAGE_TAGS

        if "AsymCoinGame" in hp["env_name"]:
            hp["x_limits"] = (-0.5, 3.0)
            hp["y_limits"] = (-1.1, 0.6)
            hp["env_class"] = vectorized_coin_game.AsymVectorizedCoinGame
        elif "MixedMotiveCoinGame" in hp["env_name"]:
            hp["x_limits"] = (-0.5, 1.0)
            hp["y_limits"] = (-0.5, 1.0)
            hp["env_class"] = mixed_motive_coin_game.MixedMotiveCoinGame
            hp["both_players_can_pick_the_same_coin"] = True
        else:
            hp["x_limits"] = (-0.5, 0.6)
            hp["y_limits"] = (-0.5, 0.6)
            hp["env_class"] = vectorized_coin_game.VectorizedCoinGame
    else:
        hp["bs_epi_mul"] = 1
        hp["bs_epi_mul_spl"] = hp["bs_epi_mul"]
        hp["spl_lr_mul"] = 3.0
        hp["base_lr"] = 0.09
        hp["n_steps_per_epi"] = 20
        hp["buf_frac"] = 0.5
        hp["n_epi"] = 10 if hp["debug"] else 800
        hp["sgd_momentum"] = 0.0
        hp["last_exploration_temp_value"] = 0.1

        hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.50), 0.5),
                (
                    int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.75),
                    hp["last_exploration_temp_value"],
                ),
            ],
            outside_value=hp["last_exploration_temp_value"],
            framework="torch",
        )

        hp["plot_keys"] += matrix_sequential_social_dilemma.PLOT_KEYS
        hp[
            "plot_assemblage_tags"
        ] += matrix_sequential_social_dilemma.PLOT_ASSEMBLAGE_TAGS

        if "IteratedPrisonersDilemma" in hp["env_name"]:
            hp["x_limits"] = (-3.5, 0.5)
            hp["y_limits"] = (-3.5, 0.5)
            hp["utilitarian_filtering_threshold"] = -2.5
            hp[
                "env_class"
            ] = matrix_sequential_social_dilemma.IteratedPrisonersDilemma
        elif "IteratedAsymBoS" in hp["env_name"]:
            hp["x_limits"] = (-0.1, 4.1)
            hp["y_limits"] = (-0.1, 4.1)
            hp["utilitarian_filtering_threshold"] = 3.2
            hp["env_class"] = matrix_sequential_social_dilemma.IteratedAsymBoS
        else:
            raise NotImplementedError(f'hp["env_name"]: {hp["env_name"]}')

    hp["lr_schedule"] = [
        (0, 0.0),
        (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.05), hp["base_lr"]),
        (int(hp["n_steps_per_epi"] * hp["n_epi"]), hp["base_lr"] / 1e9),
    ]

    hp["plot_axis_scale_multipliers"] = (
        (1 / hp["n_steps_per_epi"]),  # for x axis
        (1 / hp["n_steps_per_epi"]),
    )  # for y axis

    return hp


def _get_rllib_config(hp: dict):
    stop = {
        "episodes_total": hp["n_epi"],
    }

    env_config = _get_env_config(hp)

    my_uncertain_env_class = add_RewardUncertaintyEnvClassWrapper(
        hp["env_class"], reward_uncertainty_std=hp["reward_uncertainty_std"]
    )

    rllib_config = copy.deepcopy(ltft.DEFAULT_CONFIG)
    rllib_config.update(
        {
            "env": my_uncertain_env_class,
            "env_config": env_config,
            "multiagent": {
                "policies": {
                    env_config["players_ids"][0]: (
                        None,
                        hp["env_class"]({}).OBSERVATION_SPACE,
                        hp["env_class"].ACTION_SPACE,
                        {},
                    ),
                    env_config["players_ids"][1]: (
                        None,
                        hp["env_class"]({}).OBSERVATION_SPACE,
                        hp["env_class"].ACTION_SPACE,
                        {},
                    ),
                },
                "policy_mapping_fn": lambda agent_id: agent_id,
                # When replay_mode=lockstep, RLlib will replay all the agent
                # transitions at a particular timestep together in a batch.
                # This allows the policy to implement differentiable shared
                # computations between agents it controls at that timestep. When
                # replay_mode=independent,
                # transitions are replayed independently per policy.
                # "replay_mode": "lockstep",
                "observation_fn": ltft.observation_fn,
            },
            # === DQN Models ===
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 30 * hp["n_steps_per_epi"],
            # === Replay buffer ===
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": max(
                int(hp["n_steps_per_epi"] * hp["n_epi"] * hp["buf_frac"]), 5
            ),
            # Whether to use dueling dqn
            "dueling": False,
            # Dense-layer setup for each the advantage branch and the value branch
            # in a dueling architecture.
            "hiddens": hp["hiddens"],
            # Whether to use double dqn
            "double_q": True,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": False,
            "model": {
                # Number of hidden layers for fully connected net
                "fcnet_hiddens": hp["hiddens"],
                # Nonlinearity for fully connected net (tanh, relu)
                "fcnet_activation": "relu",
            },
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
                "type": exploration.SoftQScheduleWtClustering,
                # Add constructor kwargs here (if any).
                "temperature_schedule": hp["temperature_schedule"],
                "clustering_distance": hp["clustering_distance"],
            },
            "gamma": hp["gamma"],
            # Minimum env steps to optimize for per train call. This value does
            # not affect learning, only the length of iterations.
            "timesteps_per_iteration": hp["n_steps_per_epi"]
            if hp["debug"]
            else int(hp["n_steps_per_epi"] * hp["n_epi"] / hp["log_n_points"]),
            "min_iter_time_s": 0.0,
            "seed": tune.grid_search(hp["seeds"]),
            # === Optimization ===
            "optimizer": {
                "sgd_momentum": hp["sgd_momentum"],
            },
            # Learning rate for adam optimizer
            "lr": hp["base_lr"],
            # Learning rate schedule
            "lr_schedule": hp["lr_schedule"],
            # If not None, clip gradients during optimization at this value
            "grad_clip": 1,
            # How many steps of the model to sample before learning starts.
            "learning_starts": int(hp["n_steps_per_epi"] * hp["bs_epi_mul"]),
            # Update the replay buffer with this many samples at once. Note that
            # this setting applies per-worker if num_workers > 1.
            "rollout_fragment_length": hp["n_steps_per_epi"],
            "training_intensity": hp["training_intensity"],
            # Size of a batch sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": int(hp["n_steps_per_epi"] * hp["bs_epi_mul"]),
            # General config
            "framework": "torch",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            # LTFTTorchPolicy supports only 1 worker only otherwise
            # it would be mixing several opponents trajectories
            "num_workers": 0,
            # LTFTTorchPolicy supports only 1 env per worker only
            # otherwise several episodes would be played at the same
            # time
            "num_envs_per_worker": 1,
            "batch_mode": "complete_episodes",
            "logger_config": {
                "wandb": {
                    "project": "LTFT",
                    "group": hp["exp_name"],
                    "api_key_file": os.path.join(
                        os.path.dirname(__file__), "../../../api_key_wandb"
                    ),
                    "log_config": True,
                },
            },
            # === Debug Settings ===
            "log_level": "INFO",
            # Callbacks that will be run during various phases of training. See the
            # `DefaultCallbacks` class and
            # `examples/custom_metrics_and_callbacks.py`
            # for more usage information.
            "callbacks": callbacks.merge_callbacks(
                ltft.LTFTCallbacks,
                log.get_logging_callbacks_class(
                    log_full_epi=True,
                ),
            ),
        }
    )

    hp, rllib_config, env_config, stop = _modify_config_for_coin_game(
        hp, rllib_config, env_config, stop
    )

    nested_policies_config = rllib_config["nested_policies"]
    nested_spl_policy_config = nested_policies_config[3]["config_update"]
    nested_spl_policy_config["train_batch_size"] = (
        int(hp["n_steps_per_epi"] * hp["bs_epi_mul_spl"]),
    )
    rllib_config["nested_policies"] = nested_policies_config

    return rllib_config, env_config, stop


def _get_env_config(hp):
    if hp["env_name"] in ("IteratedPrisonersDilemma",):
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hp["n_steps_per_epi"],
        }
    elif "CoinGame" in hp["env_name"]:
        env_config = {
            "grid_size": 3,
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hp["n_steps_per_epi"],
        }
    else:
        raise NotImplementedError()

    return env_config


def _modify_config_for_coin_game(hp, rllib_config, env_config, stop):
    if "CoinGame" in hp["env_name"]:
        rllib_config.update(
            {
                "model": {
                    "dim": env_config["grid_size"],
                    # [Channel, [Kernel, Kernel], Stride]]
                    "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
                }
            }
        )

    return hp, rllib_config, env_config, stop


def _train_in_self_play(rllib_config, stop, exp_name, hp):
    selfplay_sub_dir = "self_play"
    full_exp_name = os.path.join(exp_name, selfplay_sub_dir)

    tune_analysis_self_play = ray.tune.run(
        ltft.LTFTTrainer,
        config=rllib_config,
        verbose=1,
        stop=stop,
        checkpoint_at_end=True,
        name=full_exp_name,
        log_to_file=not hp["debug"],
        loggers=None if hp["debug"] else DEFAULT_LOGGERS + (WandbLogger,),
    )

    if not hp["debug"]:
        aggregate_and_plot_tensorboard_data.add_summary_plots(
            main_path=os.path.join("~/ray_results/", full_exp_name),
            plot_keys=hp["plot_keys"],
            plot_assemble_tags_in_one_plot=hp["plot_assemblage_tags"],
        )

    return tune_analysis_self_play


def _train_against_opponent(hp, rllib_config, stop, exp_name, env_config):
    rllib_config = _modify_config_for_play_agaisnt_opponent(
        rllib_config, env_config, hp
    )

    against_opp_sub_dir = "against_opp"
    full_exp_name = os.path.join(exp_name, against_opp_sub_dir)

    tune_analysis_naive_opponent = ray.tune.run(
        ltft.LTFTTrainer,
        config=rllib_config,
        verbose=1,
        stop=stop,
        checkpoint_at_end=True,
        name=full_exp_name,
        log_to_file=not hp["debug"],
        loggers=None if hp["debug"] else DEFAULT_LOGGERS + (WandbLogger,),
    )

    if not hp["debug"]:
        aggregate_and_plot_tensorboard_data.add_summary_plots(
            main_path=os.path.join("~/ray_results/", full_exp_name),
            plot_keys=ltft.PLOT_KEYS + hp["plot_keys"],
            plot_assemble_tags_in_one_plot=ltft.PLOT_ASSEMBLAGE_TAGS
            + hp["plot_assemblage_tags"],
        )

    return tune_analysis_naive_opponent


def _modify_config_for_play_agaisnt_opponent(rllib_config, env_config, hp):
    if hp["against_evader_exploiter"] is not None:
        rllib_config = _set_config_to_use_exploiter(
            rllib_config, env_config, hp
        )
    else:
        rllib_config = _set_config_to_use_naive_opponent(
            rllib_config, env_config, hp
        )

    return rllib_config


def _set_config_to_use_exploiter(rllib_config, env_config, hp):
    exploiter_hp = hp["against_evader_exploiter"]
    n_steps_during_training = hp["n_epi"] * hp["n_steps_per_epi"]

    MyCoopDQNTorchPolicy = augmented_dqn.MyDQNTorchPolicy.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.welfares_postprocessing_fn(),
            postprocess_nstep_and_prio,
        )
    )
    exploiter_policy_config = {
        "copy_weights_every_n_steps": exploiter_hp["copy_weights_delay"]
        * n_steps_during_training,
        "start_exploit_at_step_n": exploiter_hp["start_exploit"]
        * n_steps_during_training,
        "welfare_key": postprocessing.WELFARE_UTILITARIAN,
        "nested_policies": [
            # You need to provide the policy class for every nested Policies
            {
                "Policy_class": MyCoopDQNTorchPolicy,
                "config_update": {
                    postprocessing.ADD_UTILITARIAN_WELFARE: True
                },
            },
            {
                "Policy_class": augmented_dqn.MyDQNTorchPolicy,
                "config_update": {},
            },
        ],
    }

    rllib_config["multiagent"]["policies"][env_config["players_ids"][1]] = (
        InfluenceEvaderTorchPolicy,
        hp["env_class"]().OBSERVATION_SPACE,
        hp["env_class"].ACTION_SPACE,
        exploiter_policy_config,
    )

    return rllib_config


def _set_config_to_use_naive_opponent(rllib_config, env_config, hp):
    rllib_config["multiagent"]["policies"][env_config["players_ids"][1]] = (
        augmented_dqn.MyDQNTorchPolicy,
        hp["env_class"]().OBSERVATION_SPACE,
        hp["env_class"].ACTION_SPACE,
        {},
    )
    return rllib_config


if __name__ == "__main__":
    debug_mode = True
    main(debug=debug_mode)
