import copy
import os

import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.tune.utils import validate_save_restore

torch, nn = try_import_torch()

from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma
from marltoolbox.algos import amTFT
from marltoolbox.utils.exploration import SoftQSchedule
from marltoolbox.utils import experimenting

# TODO update the unique rollout worker after every episode
# TODO check than no bug arise from the fact that there is 2 policies
#  (one used to produce samples in the rolloutworker and one used to train the models)
if __name__ == "__main__":
    n_epi = int(200 * 1.0)
    n_steps_per_epi = 20
    bs_epi_mul = 4
    base_lr = 0.04

    stop = {
        "episodes_total": n_epi,  # 4000 steps in 200 epi
    }

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": n_steps_per_epi,
        "reward_randomness": 0.1,
    }

    amTFT_config_update = merge_dicts(
        amTFT.AMTF_DEFAULT_CONFIG_UPDATE,
        {
            # Set to True to train the nested policies and to False to use them
            "amTFT_train": True,
        }
    )

    # TODO remove the useless hyper-parameters
    trainer_config_update = {
        "env": IteratedPrisonersDilemma,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "player_row": (
                    # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
                    amTFT.amTFTTorchPolicy,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    copy.deepcopy(amTFT_config_update)),
                "player_col": (
                    amTFT.amTFTTorchPolicy,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    copy.deepcopy(amTFT_config_update)),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        # === DQN Models ===
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": n_steps_per_epi,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": n_steps_per_epi,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(n_steps_per_epi * n_epi),
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
        # Can't restaure stuff with search
        "seed": tune.grid_search([1,]),
        # "seed": 1,

        "evaluation_num_episodes": 100,
        "evaluation_interval": n_epi,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": base_lr,
        # Learning rate schedule
        "lr_schedule": [(0, base_lr), (int(n_steps_per_epi * n_epi), base_lr / 1e9)],
        # Adam epsilon hyper parameter
        # "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": 1,
        # How many steps of the model to sample before learning starts.
        "learning_starts": int(n_steps_per_epi * bs_epi_mul),
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": n_steps_per_epi,
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": int(n_steps_per_epi * bs_epi_mul),

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
            "type": SoftQSchedule,
            # Add constructor kwargs here (if any).
            "temperature_schedule": PiecewiseSchedule(
                endpoints=[
                    (0, 1.0), (int(n_steps_per_epi * n_epi * 0.75), 0.1)],
                outside_value=0.5,
                framework="torch")
        },

        # General config
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # LE supports only 1 worker only otherwise it would be mixing several opponents trajectories
        "num_workers": 0,
        # LE supports only 1 env per worker only otherwise several episodes would be played at the same time
        "num_envs_per_worker": 1,

        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
        # for more usage information.
        # "callbacks": DefaultCallbacks,
        "callbacks": amTFT.amTFTCallBacks,

    }

    # Train

    ray.init(num_cpus=5, num_gpus=0)

    results = ray.tune.run(amTFT.amTFTTrainer, config=trainer_config_update,
                           stop=stop, verbose=1, name="amTFT_self_play_IPD",
                           checkpoint_at_end=True,
                           metric="episode_reward_mean", mode="max")


    # Evaluate

    for policy_id in trainer_config_update["multiagent"]["policies"].keys():
        trainer_config_update["multiagent"]["policies"][policy_id][3][amTFT.AMTFT_TRAIN_KEY] = False
    trainer_config_update["seed"] = 3

    stop = {"episodes_total": 100}

    experiment = experimenting.SameAndCrossPlay(TrainerClass=amTFT.amTFTTrainer,
                                                extract_checkpoints_from_results=results,
                                                evaluation_config=trainer_config_update,
                                                stop_config=stop,
                                                exp_name="amTFT_self_play_IPD_eval",
                                                n_same_play_per_checkpoint=1,
                                                n_cross_play_per_checkpoint=0)

    ray.shutdown()
