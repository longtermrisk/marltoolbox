##########
# Code modified from: https://github.com/julianstastny/implicit-bargaining-problems/blob/master/inequity_aversion.py
##########

import ray
from ray import tune

from marltoolbox.algos.inequity_aversion import InequityAversionTrainer
from marltoolbox.envs.matrix_SSD import IteratedBoSAndPD

def main(debug):
    ray.init(num_cpus=5, num_gpus=0)

    stop = {"episodes_total": 10 if debug else 10000}

    env_config = {
        "max_steps": 10,
        "players_ids": ["player_row", "player_col"],
    }

    policies = {env_config["players_ids"][0]: (None, IteratedBoSAndPD.OBSERVATION_SPACE, IteratedBoSAndPD.ACTION_SPACE, {}),
                env_config["players_ids"][1]: (None, IteratedBoSAndPD.OBSERVATION_SPACE, IteratedBoSAndPD.ACTION_SPACE, {})}

    trainer_config_update = {
        "env": IteratedBoSAndPD,
        "env_config": env_config,
        # General
        "num_gpus": 0,
        "num_workers": 1,
        # Method specific
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: agent_id),
        },
        "framework": "torch",
    }

    tune.run(InequityAversionTrainer, stop=stop, checkpoint_freq=10, config=trainer_config_update)
    ray.shutdown()

if __name__ == "__main__":
    debug = True
    main(debug)

