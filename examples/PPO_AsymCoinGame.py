import argparse
import os

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from marltoolbox.envs.coin_game import CoinGame

parser = argparse.ArgumentParser()
parser.add_argument("--tf", action="store_true")
parser.add_argument("--stop-iters", type=int, default=2000)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    stop = {
        "training_iteration": args.stop_iters,
    }

    env_config = {
        "players_ids": ["player_red", "player_blue"],
        "max_steps": 20,
        "reward_randomness": 0.0,
        "grid_size": 3,
        "get_additional_info": True,
        "asymmetric": True,
    }

    trainer_config_update = {
        "env": CoinGame,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (None,
                               CoinGame(env_config).OBSERVATION_SPACE,
                               CoinGame.ACTION_SPACE,
                               {
                                   "framework": "tf" if args.tf else "torch",
                               }),
                env_config["players_ids"][1]: (None,
                               CoinGame(env_config).OBSERVATION_SPACE,
                               CoinGame.ACTION_SPACE,
                               {
                                   "framework": "tf" if args.tf else "torch",
                               }),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "framework": "tf" if args.tf else "torch",

        "model": {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]]  # [Channel, [Kernel, Kernel], Stride]]
          },


        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "log_level": "INFO"
    }

    results = tune.run(PPOTrainer, config=trainer_config_update, stop=stop, verbose=1)
    ray.shutdown()
