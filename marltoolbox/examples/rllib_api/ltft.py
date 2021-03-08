import copy
import os

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn_torch_policy import postprocess_nstep_and_prio
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule

torch, nn = try_import_torch()

from marltoolbox.envs.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma
from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame
from marltoolbox.algos import ltft
from marltoolbox.algos.ltft.utils import MyDQNTorchPolicy
from marltoolbox.utils import log, miscellaneous, exploration, postprocessing
from marltoolbox.envs.utils.wrappers import \
    add_RewardUncertaintyEnvClassWrapper
from marltoolbox.examples.rllib_api.amtft_various_env import \
    modify_hyperparams_for_the_selected_env
from marltoolbox.algos.exploiters.influence_evader import \
    InfluenceEvaderTorchPolicy
from marltoolbox.utils.postprocessing import ADD_UTILITARIAN_WELFARE


def main(debug, env=None, train_n_replicates=None):
    hparameters, exp_name = get_hyparameters(debug, env, train_n_replicates)

    rllib_config, env_config, stop = get_rllib_config(hparameters, debug)

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)

    tune_analysis_self_play = train_in_self_play(
        rllib_config, stop, exp_name)

    tune_analysis_naive_opponent = None
    # tune_analysis_naive_opponent = train_against_opponent(
    #     hparameters, rllib_config, stop, exp_name, env_config)

    ray.shutdown()
    return tune_analysis_self_play, tune_analysis_naive_opponent


def get_hyparameters(debug, env=None, train_n_replicates=None):
    if debug:
        train_n_replicates = 1
    elif train_n_replicates is None:
        train_n_replicates = 1

    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("LTFT")

    hparameters = {
        "n_epi": 10 if debug else 200,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "base_lr": 0.04,
        "spl_lr_mul": 10.0,
        "seeds": seeds,
        "debug": debug,
        "hiddens": [64],
        "buf_frac": 0.15,

        "env": "IteratedPrisonersDilemma",
        # "env": "CoinGame",
        "reward_uncertainty_std": 0.0,  # 0.1,

        # "against_evader_exploiter": None,
        "against_evader_exploiter": {
            "start_exploit": 0.50,
            "copy_weights_delay": 0.10,
        },

    }

    if env is not None:
        hparameters["env"] = env

    hparameters = modify_hyperparams_for_the_selected_env(hparameters)

    hparameters["gamma"] = 0.5
    hparameters["last_exploration_temp_value"] = 0.2
    hparameters["temperature_schedule"] = PiecewiseSchedule(
        endpoints=[
            (0, 2.0),
            (int(hparameters["n_steps_per_epi"] * hparameters["n_epi"] *
                 0.50), 0.2),
            (int(hparameters["n_steps_per_epi"] * hparameters["n_epi"] * 0.75),
             hparameters["last_exploration_temp_value"])],
        outside_value=hparameters["last_exploration_temp_value"],
        framework="torch")

    return hparameters, exp_name


def get_rllib_config(hp: dict, debug):
    stop = {
        "episodes_total": hp["n_epi"],
    }

    env_config = get_env_config(hp)

    ltft_config = ltft.prepare_default_config(
        lr=hp["base_lr"],
        lr_spl=hp["base_lr"] * hp["spl_lr_mul"],
        n_epi=hp["n_epi"],
        n_steps_per_epi=hp["n_steps_per_epi"])

    my_uncertain_env_class = add_RewardUncertaintyEnvClassWrapper(
        hp["env"],
        reward_uncertainty_std=hp["reward_uncertainty_std"])

    rllib_config = copy.deepcopy(ltft_config)
    rllib_config.update({
        "env": my_uncertain_env_class,
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
        "buffer_size": int(hp["n_steps_per_epi"] * hp["n_epi"] *
                           hp["buf_frac"]),
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

        "gamma": hp["gamma"],
        "min_iter_time_s":
            0.0 if debug else
            10.0 if hp["env"] in (CoinGame, AsymCoinGame) else 1.0,
        "seed": tune.grid_search(hp["seeds"]),

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": hp["base_lr"],
        # Learning rate schedule
        "lr_schedule": [
            (0, hp["base_lr"]),
            (int(hp["n_steps_per_epi"] * hp["n_epi"]), hp["base_lr"] / 1e9)],
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
        # `DefaultCallbacks` class and
        # `examples/custom_metrics_and_callbacks.py`
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
                # The Exploration class to use. In the simplest case,
                # this is the name (str) of any class present in the
                # `rllib.utils.exploration` package.
                # You can also provide the python class directly or
                # the full location of your class (e.g.
                # "ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy").
                # "type": exploration.SoftQSchedule,
                "type": exploration.SoftQScheduleWtClustering,
                # Add constructor kwargs here (if any).
                "temperature_schedule": hp["temperature_schedule"],
                "clustering_distance": 0.5,
            },
            "model": {
                "dim": env_config["grid_size"],
                # [Channel, [Kernel, Kernel], Stride]]
                "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
            }
        })

        nested_policies_config = rllib_config["nested_policies"]
        nested_spl_policy_config= nested_policies_config[3]["config_update"]
        nested_spl_policy_config["sgd_momentum"] = 0.9
        rllib_config["nested_policies"] = nested_policies_config

    return hp, rllib_config, env_config, stop


def train_in_self_play(rllib_config, stop, exp_name):
    tune_analysis_self_play = ray.tune.run(
        ltft.LTFTTrainer, config=rllib_config,
        verbose=1, checkpoint_freq=0, stop=stop,
        checkpoint_at_end=True, name=exp_name)
    return tune_analysis_self_play


def train_against_opponent(hp, rllib_config, stop, exp_name, env_config):
    rllib_config = modify_config_for_play_agaisnt_opponent(
        rllib_config, env_config, hp)

    tune_analysis_naive_opponent = ray.tune.run(
        ltft.LTFTTrainer,
        config=rllib_config,
        verbose=1, checkpoint_freq=0, stop=stop,
        checkpoint_at_end=True, name=exp_name)
    return tune_analysis_naive_opponent


def modify_config_for_play_agaisnt_opponent(rllib_config, env_config, hp):
    if hp["against_evader_exploiter"] is not None:
        rllib_config = set_config_to_use_evader_exploiter(
            rllib_config, env_config, hp)
    else:
        rllib_config = set_config_to_use_naive_opponent(
            rllib_config, env_config, hp)

    return rllib_config


def set_config_to_use_evader_exploiter(rllib_config, env_config, hp):
    exploiter_hp = hp["against_evader_exploiter"]
    n_steps_during_training = hp["n_epi"] * hp["n_steps_per_epi"]

    MyCoopDQNTorchPolicy = MyDQNTorchPolicy.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.welfares_postprocessing_fn(),
            postprocess_nstep_and_prio
        )
    )
    exploiter_policy_config = {
        "copy_weights_every_n_steps":
            exploiter_hp["copy_weights_delay"] * n_steps_during_training,
        "start_exploit_at_step_n":
            exploiter_hp["start_exploit"] * n_steps_during_training,
        "welfare_key": postprocessing.WELFARE_UTILITARIAN,
        'nested_policies': [
            # You need to provide the policy class for every nested Policies
            {"Policy_class": MyCoopDQNTorchPolicy,
             "config_update": {ADD_UTILITARIAN_WELFARE: True}},
            {"Policy_class": MyDQNTorchPolicy,
             "config_update": {}}
        ],
    }

    rllib_config["multiagent"]["policies"][env_config["players_ids"][1]] = (
        InfluenceEvaderTorchPolicy,
        hp["env"]().OBSERVATION_SPACE,
        hp["env"].ACTION_SPACE,
        exploiter_policy_config
    )

    return rllib_config


def set_config_to_use_naive_opponent(rllib_config, env_config, hp):
    rllib_config["multiagent"]["policies"][env_config["players_ids"][1]] = (
        dqn.DQNTorchPolicy,
        hp["env"]().OBSERVATION_SPACE,
        hp["env"].ACTION_SPACE,
        {}
    )
    return rllib_config


if __name__ == "__main__":
    debug_mode = True
    main(debug=debug_mode)
