import argparse
import os
import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer

from marltoolbox.envs.matrix_sequential_social_dilemma import IteratedPrisonersDilemma
from marltoolbox.utils import log, miscellaneous

def main(debug, stop_iters=300, tf=False):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("PG_IPD")

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)

    rllib_config, stop_config = get_rllib_config(seeds, debug, stop_iters, tf)
    tune_analysis = tune.run(PGTrainer, config=rllib_config, stop=stop_config,
                             checkpoint_freq=0, checkpoint_at_end=True, name=exp_name)
    ray.shutdown()
    return tune_analysis


def get_rllib_config(seeds, debug=False, stop_iters=300, tf=False):
    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": 20,
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
                    {}
                ),
                env_config["players_ids"][1]: (
                    None,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    {}
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        "seed": tune.grid_search(seeds),
        "callbacks": log.get_logging_callbacks_class(),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "tf" if tf else "torch",
    }

    return rllib_config, stop_config


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
