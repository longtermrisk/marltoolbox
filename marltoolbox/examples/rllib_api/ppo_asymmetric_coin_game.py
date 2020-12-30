import argparse
import os
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from marltoolbox.envs.coin_game import CoinGame
from marltoolbox.utils import log, miscellaneous

parser = argparse.ArgumentParser()
parser.add_argument("--tf", action="store_true")
parser.add_argument("--stop-iters", type=int, default=2000)


def main(stop_iters, tf, debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("PPO_AsymCG")

    ray.init()

    stop = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {
        "players_ids": ["player_red", "player_blue"],
        "max_steps": 20,
        "reward_randomness": None,
        "grid_size": 3,
        "get_additional_info": True,
        "asymmetric": True,
    }

    rllib_config = {
        "env": CoinGame,
        "env_config": env_config,

        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (None,
                                               CoinGame(env_config).OBSERVATION_SPACE,
                                               CoinGame.ACTION_SPACE,
                                               {
                                                   "framework": "tf" if tf else "torch",
                                               }),
                env_config["players_ids"][1]: (None,
                                               CoinGame(env_config).OBSERVATION_SPACE,
                                               CoinGame.ACTION_SPACE,
                                               {
                                                   "framework": "tf" if tf else "torch",
                                               }),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        # Size of batches collected from each worker.
        "rollout_fragment_length": 20 if debug else 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 128 if debug else 4000,
        "model": {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]]  # [Channel, [Kernel, Kernel], Stride]]
        },

        "seed": tune.grid_search(seeds),
        "callbacks": log.get_logging_callbacks_class(),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "log_level": "INFO",
        "framework": "tf" if tf else "torch",
    }

    results = tune.run(PPOTrainer, config=rllib_config, stop=stop, verbose=1,
                       checkpoint_freq=0, checkpoint_at_end=True, name=exp_name)
    ray.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    debug_mode = True
    main(args.stop_iters, args.tf, debug_mode)
