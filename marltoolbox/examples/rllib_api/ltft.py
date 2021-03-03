import copy
import os

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTorchPolicy
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from marltoolbox.envs.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma
from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame
from marltoolbox.algos import ltft
from marltoolbox.utils import log, miscellaneous, exploration
from marltoolbox.envs.utils.wrappers import add_RewardUncertaintyEnvClassWrapper
from marltoolbox.examples.rllib_api.amtft_various_env import \
    modify_hyperparams_for_the_selected_env


def main(debug):

    hparameters, exp_name = get_hyparameters(debug)

    rllib_config, env_config, stop = get_rllib_config(hparameters)

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=True)

    tune_analysis_self_play = train_in_self_play(
        rllib_config, stop, exp_name)

    tune_analysis_naive_opponent = train_against_naive_opponent(
        hparameters, rllib_config, stop, exp_name, env_config)

    ray.shutdown()
    return tune_analysis_self_play, tune_analysis_naive_opponent

def get_hyparameters(debug, env=None):
    train_n_replicates = 1 if debug else 10
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("LTFT_IPD")

    hparameters = {
        "n_epi": 10 if debug else 200,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "base_lr": 0.04,
        "spl_lr_mul": 10.0,
        "seeds": seeds,
        "debug": debug,

        # "env": "IteratedPrisonersDilemma",
        "env": "CoinGame",
    }

    if env is not None:
        hparameters["env"] = env

    hparameters = modify_hyperparams_for_the_selected_env(hparameters)

    return hparameters, exp_name

def get_rllib_config(hp: dict):
    stop = {
        "episodes_total": hp["n_epi"],
    }

    env_config = get_env_config(hp)

    ltft_config = ltft.prepare_default_config(
        lr=hp["base_lr"],
        lr_spl=hp["base_lr"] * hp["spl_lr_mul"],
        n_epi=hp["n_epi"],
        n_steps_per_epi=hp["n_steps_per_epi"])

    MyUncertainEnv = add_RewardUncertaintyEnvClassWrapper(
        hp["env"],
        reward_uncertainty_std=0.1)

    rllib_config = copy.deepcopy(ltft_config)
    rllib_config.update({
        "env": MyUncertainEnv,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    None,
                    hp["env"]({}).OBSERVATION_SPACE,
                    hp["env"].ACTION_SPACE,
                    {}),
                env_config["players_ids"][1]: (
                    None,
                    hp["env"]({}).OBSERVATION_SPACE,
                    hp["env"].ACTION_SPACE,
                    {}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        # === DQN Models ===
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": hp["n_steps_per_epi"],
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": hp["n_steps_per_epi"],
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(hp["n_steps_per_epi"] * hp["n_epi"]),
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [4],
        # Whether to use double dqn
        "double_q": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        "model": {
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": [4, 2],
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "relu",
        },

        "gamma": 0.5,
        "min_iter_time_s": 0.33,
        "seed": tune.grid_search(hp["seeds"]),

        # === Optimization ===
        # Adam epsilon hyper parameter
        # "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": 1,
        # How many steps of the model to sample before learning starts.
        "learning_starts": int(hp["n_steps_per_epi"] * hp["bs_epi_mul"]),
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": hp["n_steps_per_epi"],
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

        # === Debug Settings ===
        "log_level": "INFO",
        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
        # for more usage information.
        "callbacks": miscellaneous.merge_callbacks(
            ltft.LTFTCallbacks,
            log.get_logging_callbacks_class()),
    })

    hp, rllib_config, env_config, stop = \
        modify_config_for_coin_game(hp, rllib_config, env_config, stop)

    return rllib_config, env_config, stop


def get_env_config(hp):
    if hp["env"] in (IteratedPrisonersDilemma,):
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hp["n_steps_per_epi"],
        }

    elif hp["env"] in (CoinGame, AsymCoinGame):
        env_config = {
            "grid_size": 3,
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hp["n_steps_per_epi"],
        }
    else:
        raise NotImplementedError()

    return env_config

def modify_config_for_coin_game(hp, rllib_config, env_config, stop):
    if hp["env"] in (CoinGame, AsymCoinGame):

        rllib_config.update({
            # === Exploration Settings ===
            # Default exploration behavior, iff `explore`=None is passed into
            # compute_action(s).
            # Set to False for no exploration behavior (e.g., for evaluation).
            "explore": True,
            # Provide a dict specifying the Exploration object's config.
            "exploration_config": {
                # The Exploration class to use. In the simplest case, this is the name
                # (str) of any class present in the `rllib.utils.exploration` package.
                # You can also provide the python class directly or the full location
                # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
                # EpsilonGreedy").
                "type": exploration.SoftQSchedule,
                # Add constructor kwargs here (if any).
                "temperature_schedule": hp["temperature_schedule"],
            },
            "model": {
                "dim": env_config["grid_size"],
                # [Channel, [Kernel, Kernel], Stride]]
                "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
            }
        })

    return hp, rllib_config, env_config, stop


def train_in_self_play(rllib_config, stop, exp_name):
    tune_analysis_self_play = ray.tune.run(
        ltft.LTFTTrainer, config=rllib_config,
        verbose=1, checkpoint_freq=0, stop=stop,
        checkpoint_at_end=True, name=exp_name)
    return tune_analysis_self_play


def train_against_naive_opponent(hp, rllib_config, stop, exp_name, env_config):
    # Set player_col to use a naive policy
    rllib_config["multiagent"]["policies"][env_config["players_ids"][1]] = (
        DQNTorchPolicy,
        hp["env"]().OBSERVATION_SPACE,
        hp["env"].ACTION_SPACE,
        {}
    )
    tune_analysis_naive_opponent = ray.tune.run(
        ltft.LTFTTrainer,
        config=rllib_config,
        verbose=1, checkpoint_freq=0, stop=stop,
        checkpoint_at_end=True, name=exp_name)
    return tune_analysis_naive_opponent


if __name__ == "__main__":
    debug = True
    main(debug)
