import argparse
import os
import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer

from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma
from marltoolbox.utils import log, miscellaneous

parser = argparse.ArgumentParser()
parser.add_argument("--tf", action="store_true")
parser.add_argument("--stop-iters", type=int, default=200)


def main(stop_iters, tf, debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("PG_IPD")

    ray.init(num_cpus=os.cpu_count(), num_gpus=0)

    stop = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": 20,
        "reward_randomness": None,
        "get_additional_info": True,
    }

    rllib_config = {
        "env": IteratedPrisonersDilemma,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    None,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    {
                        "framework": "tf" if tf else "torch",
                    }
                ),
                env_config["players_ids"][1]: (
                    None,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    {
                        "framework": "tf" if tf else "torch",
                    }
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        "seed": tune.grid_search(seeds),
        "callbacks": log.get_logging_callbacks_class(),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "log_level": "INFO",
        "framework": "tf" if tf else "torch",
    }

    results = tune.run(PGTrainer, config=rllib_config, stop=stop, verbose=1,
                       checkpoint_freq=0, checkpoint_at_end=True, name=exp_name)
    ray.shutdown()


if __name__ == "__main__":
    debug_mode = True
    args = parser.parse_args()
    main(args.stop_iters, args.tf, debug_mode)
