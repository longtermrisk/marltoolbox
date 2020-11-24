import argparse
import os

import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer

from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma

parser = argparse.ArgumentParser()
parser.add_argument("--tf", action="store_true")
parser.add_argument("--stop-iters", type=int, default=200)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=5, num_gpus=0)

    stop = {
        "training_iteration": args.stop_iters,
    }

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": 20,
        "reward_randomness": 0.0,
        "get_additional_info": True,
    }

    trainer_config_update = {
        "env": IteratedPrisonersDilemma,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (None,
                               IteratedPrisonersDilemma.OBSERVATION_SPACE,
                               IteratedPrisonersDilemma.ACTION_SPACE,
                               {
                                   "framework": "tf" if args.tf else "torch",
                               }),
                env_config["players_ids"][1]: (None,
                               IteratedPrisonersDilemma.OBSERVATION_SPACE,
                               IteratedPrisonersDilemma.ACTION_SPACE,
                               {
                                   "framework": "tf" if args.tf else "torch",
                               }),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "framework": "tf" if args.tf else "torch",

        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "log_level": "INFO"
    }

    results = tune.run(PGTrainer, config=trainer_config_update, stop=stop,
                       verbose=1, checkpoint_freq=30)
    ray.shutdown()
