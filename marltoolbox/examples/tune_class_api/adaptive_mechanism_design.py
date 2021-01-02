##########
# Code from: https://github.com/tobiasbaumann1/Adaptive_Mechanism_Design
##########
import copy

import logging
import math
import os
import ray
import time
from ray import tune

logging.basicConfig(filename='main.log', level=logging.DEBUG, filemode='w')
from marltoolbox.algos.adaptive_mechanism_design.adaptive_mechanism_design import AdaptiveMechanismDesign
from marltoolbox.utils import log


def train(tune_hp):
    tune_config = copy.deepcopy(tune_hp)

    stop = {
        "episodes_total": tune_hp["n_episodes"],
    }

    if tune_hp["env"] == "FearGreedMatrix":
        tune_config["env_config"] = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": tune_hp["n_steps_per_epi"],
            "reward_randomness": 0.0,
            "get_additional_info": True,
        }
    elif tune_hp["env"] == "CoinGame":
        tune_config["env_config"] = {
            "players_ids": ["player_red", "player_blue"],
            "max_steps": tune_hp["n_steps_per_epi"],
            "get_additional_info": True,
            "flatten_obs": True,
        }

    ray.init(num_cpus=1, num_gpus=1)
    training_results = tune.run(AdaptiveMechanismDesign, name=tune_hp["exp_name"], config=tune_config, stop=stop)
    ray.shutdown()
    return training_results


def add_env_hp(hp):
    if hp["env"] == "FearGreedMatrix":
        hp.update({
            "lr": 0.01,
            "gamma": 0.9,
            "n_steps_per_epi": 1,
            "n_units": 10,
            "use_simple_agents": True,
            "n_episodes": 10 if hp["debug"] else 4000,
            "max_reward_strength": 3,
            "weight_decay": 0.0,
            "convert_a_to_one_hot": False,
            "loss_mul_planner": 1.0,
            "mean_theta": -2.0,
            "std_theta": 0.5,
        })

        if hp["value_fn_variant"] == 'estimated':
            hp["mean_theta"] = -1.5
            hp["std_theta"] = 0.05

    if hp["env"] == "CoinGame":
        hp.update({
            "lr": 0.01/3.0,
            "gamma": 0.5,
            "n_steps_per_epi": 20,
            # "n_units": 8,
            # "n_units": [16, 32],
            "n_units": tune.grid_search([8, [16, 32]]),
            "use_simple_agents": False,
            "n_episodes": 10 if hp["debug"] else 16000,
            "max_reward_strength": 1.0,
            "weight_decay": 0.003,
            "convert_a_to_one_hot": True,
            "loss_mul_planner": 1.0,
            "mean_theta": None,
            "std_theta": None,
        })

    return hp


def main(debug):
    train_n_replicates = 1 if debug else 1
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("adaptive_mechanism_design")

    hyperparameters = {
        "exp_name": exp_name,
        "seed": tune.grid_search(seeds),
        "debug": debug,

        "fear": 1,

        "greed": -1,
        # "greed": 1,

        "with_redistribution": False,
        "cost_param": 0.0002,
        "n_planning_eps": math.inf,

        "value_fn_variant": 'exact',
        # "value_fn_variant": 'estimated',
        # "value_fn_variant": tune.grid_search(['exact', 'estimated']),

        "action_flip_prob": 0,
        "n_players": 2,

        "with_planner": True,
        # "with_planner": False,
        # "with_planner": tune.grid_search([True, False]),

        # "env": "FearGreedMatrix",
        "env": "CoinGame",

    }

    hyperparameters = add_env_hp(hyperparameters)

    train(hyperparameters)


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
