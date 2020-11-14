import copy
import os

import ray
from ray.rllib.agents.dqn import DQNTrainer

from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma
from marltoolbox.algos.le.learning_equilibrium import LE_DEFAULT_CONFIG_UPDATE, LE, LECallBacks

# TODO update the unique rollout worker after every episode
# TODO check than no bug arise from the fact that there is 2 policies
#  (one used to produce samples in the rolloutworker and one used to train the models)
if __name__ == "__main__":
    ray.init()

    stop = {
        "episodes_total": 300,  # 6000 steps in 300 epi
    }

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": 20,
        "reward_randomness": 0.0,
    }

    # TODO remove the useless hyper-parameters
    trainer_config_update = {
        "env": IteratedPrisonersDilemma,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "player_row": (
                    # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
                    LE,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    copy.deepcopy(LE_DEFAULT_CONFIG_UPDATE)),
                "player_col": (
                    LE,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    copy.deepcopy(LE_DEFAULT_CONFIG_UPDATE)),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        # === DQN Models ===
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": 20,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 100,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 6000,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # "gamma": 0.5,
        "model":{
            # Number of hidden layers for fully connected net
            "fcnet_hiddens":[256,2],
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "relu",
            },

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 5e-4,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": 1,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 60,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": 20,
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 60,

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
            "type": "SoftQ",
            # Add constructor kwargs here (if any).
            # TODO add SoftQ with decaying temperature
        },

        # General config
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # LE supports only 1 worker only otherwise it would be mixing several opponents trajectories
        "num_workers": 0,
        # LE supports only 1 env per worker only otherwise several episodes would be played at the same time
        "num_envs_per_worker": 1,
        "batch_mode": "complete_episodes",


        # === Debug Settings ===
        # Whether to write episode stats and videos to the agent log dir. This is
        # typically located in ~/ray_results.
        "monitor": True,
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level). When using the
        # `rllib train` command, you can also use the `-v` and `-vv` flags as
        # shorthand for INFO and DEBUG.
        "log_level": "INFO",
        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
        # for more usage information.
        # "callbacks": DefaultCallbacks,
        "callbacks": LECallBacks,
        # Whether to attempt to continue training if a worker crashes. The number
        # of currently healthy workers is reported as the "num_healthy_workers"
        # metric.
        "ignore_worker_failures": False,
        # Log system resource metrics to results. This requires `psutil` to be
        # installed for sys stats, and `gputil` for GPU metrics.
        "log_sys_usage": True,
        # Use fake (infinite speed) sampler. For testing only.
        "fake_sampler": False,
    }


    results = ray.tune.run(DQNTrainer, config=trainer_config_update,
                           stop=stop, verbose=1, name="LE_self_play_IPD")
    ray.shutdown()
