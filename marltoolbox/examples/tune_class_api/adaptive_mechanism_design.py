##########
# Code from: https://github.com/tobiasbaumann1/Adaptive_Mechanism_Design
##########
import copy
import os
import logging
import math
import ray
import time
from ray import tune
import numpy as np

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

    ray.init(num_cpus=os.cpu_count(), num_gpus=0)
    training_results = tune.run(AdaptiveMechanismDesign, name=tune_hp["exp_name"], config=tune_config, stop=stop)
    ray.shutdown()
    return training_results


def add_env_hp(hp):
    if hp["env"] == "FearGreedMatrix":
        hp.update({
            "lr": 0.01,
            "gamma": 0.9,
            "n_steps_per_epi": 1,
            # "n_steps_per_epi": 20,
            # "n_units": 10,
            "n_units": 64,
            # "use_simple_agents": True,
            "use_simple_agents": False,
            # "use_simple_agents": tune.grid_search([False, True]),
            "n_episodes": 10 if hp["debug"] else 4000,
            "max_reward_strength": 3,
            "weight_decay": 0.0,
            # "convert_a_to_one_hot": False,
            # TODO rename to lr planner scaler...
            "loss_mul_planner": 1.0,
            # "mean_theta": -2.0,
            # "std_theta": 0.5,
            # since I am using greed = 1 instead of -1 then I don't need mean = -2.0
            "mean_theta": 0.0,
            "std_theta": 0.1,
            "planner_clip_norm": None,  # no clipping
            # "planner_clip_norm": 1.0,
            "entropy_coeff": 0.0,
            # "cost_param": 0.0002/100,
            # "cost_param": 0.0,
            # "cost_param": 0.0002,
            "cost_param": 0.0002 / 2 / 2,

            "normalize_planner": True,

        })

        # if hp["value_fn_variant"] == 'estimated':
        #     hp["mean_theta"] = -1.5
        #     hp["std_theta"] = 0.05

        # if not hp["use_simple_agents"]:
        #     hp["lr"] = 0.01 / 10.0
        # #     hp["loss_mul_planner"] = 3.0
        # #     hp["cost_param"] = 0.0002

    if hp["env"] == "CoinGame":
        hp.update({
            # "lr": 0.01,
            "lr": 0.01/3.0,
            # "lr": 0.01/3.0/8.0,
            "gamma": 0.5,
            "n_steps_per_epi": 20,
            "n_units": 8, # Keep it low to make the actor learn slowly (at least with AdamOptim)
            # "cost_param": 0.0002 / 2 / 2,
            "cost_param": 0.0002 / 2 / 2 / 3,
            # "n_units": 64,
            # "cost_param": 0.0002 / 2 / 2,
            # "n_units": [16, 32],
            # "n_units": tune.grid_search([8, [16, 32]]),
            "use_simple_agents": False,
            "n_episodes": 10 if hp["debug"] else 12000,
            "max_reward_strength": 1.0,
            # "weight_decay": 0.0,
            "weight_decay": 0.003,
            # "weight_decay": 0.000003,
            # TODO remove this HP and use env name
            # "convert_a_to_one_hot": True,
            "loss_mul_planner": 1.0,
            # "loss_mul_planner": 1.0/1000,
            # "mean_theta": 0.0,
            "mean_theta": 0.0,
            "std_theta": 0.1,
            # "std_theta": 1.0,
            # "std_theta": 0.5,
            # "cost_param": 0.0,
            # "cost_param": 0.0002,
            # "cost_param": 0.0002 / 2 / 2 * 8.0,
            # "cost_param": 0.00000002,
            "planner_clip_norm": None,  # no clipping
            # "planner_clip_norm": 0.5,
            "entropy_coeff": 0.1,
            # "entropy_coeff": 0.1/3.0,
            # "entropy_coeff": 0.1 * 8.0,

            "normalize_planner": True,

            "add_state_grad": False,
            "planner_momentum": 0.9,
            "no_weights_decay_planner": True,
            "planner_std_theta_mul": 10.0,
            "use_adam_optimizer": True,
            "use_softmax_hot": False,

            # "planner_momentum": tune.grid_search([0.9, 0.99]),
            # "loss_mul_planner": tune.grid_search([3.0,1.0,1.0/3.0]),
            # "entropy_coeff": tune.grid_search([0.1,0.1/3.0]),

            # "add_state_grad": tune.grid_search([True, False]),
            # "use_adam_optimizer": tune.grid_search([True, False]),
            # "use_softmax_hot": tune.grid_search([True, False]),
            # "no_weights_decay_planner": tune.grid_search([True, False]),
            # "cost_param": tune.grid_search([0.0002 / 2 / 2 / 3,0.0002 / 2 / 2]),
            # "n_units": tune.grid_search([8, 64]),

        })

    return hp


def main(debug):
    train_n_replicates = 1 if debug else 1
    timestamp = int(time.time())
    # timestamp = 1610039542
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("adaptive_mechanism_design")

    hyperparameters = {
        "exp_name": exp_name,
        "seed": tune.grid_search(seeds),
        # "seed": 1610052221,
        "debug": debug,
        "report_every_n":10,

        "fear": 1,

        # "greed": -1,
        "greed": 1, # Selecting greed = 1 to be sure that the agents without planner learns DD
        # (needed when using the  not simple network)

        "with_redistribution": False,
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

        "add_state_grad": False,

    }

    hyperparameters = add_env_hp(hyperparameters)

    train(hyperparameters)


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
