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

logging.basicConfig(filename='main.log', level=logging.DEBUG, filemode='w')
from marltoolbox.algos.adaptive_mechanism_design.amd_using_rllib_agents import AdaptiveMechanismDesign
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

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=True)
    training_results = tune.run(AdaptiveMechanismDesign, name=tune_hp["exp_name"], config=tune_config, stop=stop)
    ray.shutdown()
    return training_results


def add_env_hp(hp):
    if hp["env"] == "FearGreedMatrix":
        hp.update({
            "lr": 0.01,
            "gamma": 0.9,
            "n_steps_per_epi": 1,
            # "n_units": 10,
            "n_units": 64,
            # "use_simple_agents": True,
            "use_simple_agents": False,
            # "use_simple_agents": tune.grid_search([False, True]),
            "n_episodes": 10 if hp["debug"] else 4000,
            "max_reward_strength": 3,
            "weight_decay": 0.0,
            # TODO rename to lr planner scaler...
            "loss_mul_planner": 1.0,
            # since I am using greed = 1 instead of -1 then I don't need mean = -2.0
            "mean_theta": 0.0,
            "std_theta": 0.1,
            "planner_clip_norm": None,  # no clipping
            "entropy_coeff": 0.0,
            "cost_param": 0.0002 / 2 / 2,

            "normalize_planner": True,






            "normalize_against_v": False,

        })

        if not hp["use_simple_agents"]:
            hp.update({
            # Good not use_simple_agents
            "add_state_grad": False,
            "no_weights_decay_planner": False,
            "weight_decay_pl_mul": 1.0,
            "use_adam_optimizer": False,
            "momentum": 0.9,
            "planner_momentum": 0.9,
            "planner_std_theta_mul": 2.5,
            "loss_mul_planner": 100.0 * 100,
            "cost_param": 0.0002 / 2 / 2 / 10 / 100,
            "use_softmax_hot": False,
            "square_cost": False,
            })

            if not hp["add_state_grad"]:
                hp.update({
                # Good add_state
                "n_steps_per_epi": 20,
                "lr": 0.01 / 3,
                "loss_mul_planner": 100.0 * 100,
                "weight_decay_pl_mul": 1.0 * 1000 * 3 * 3,
                "use_v_pl": True,
                })

    if hp["env"] == "CoinGame":
        hp.update({
            "report_every_n": 1,
            "n_episodes": 10 if hp["debug"] else 10000,

            "report_every_n": 10,
            "n_episodes": 10 if hp["debug"] else 1000000,

            "use_v_pl": True,
            "lr": 0.01/3.0,
            "gamma": 0.5,
            "n_steps_per_epi": 20,
            "use_simple_agents": False,
            "max_reward_strength": 1.0,
            "weight_decay": 0.003,
            "mean_theta": 0.0,
            "std_theta": 0.1,
            "planner_clip_norm": None,

            "normalize_planner": True,


            "use_adam_optimizer": False,
            "add_state_grad": False,
            "no_weights_decay_planner": False,
            "momentum": 0.99,
            "planner_momentum": 0.99,
            "entropy_coeff": 0.1/10/3,
            "loss_mul_planner": 100.0 * 100 * 10 / 3,
            "use_softmax_hot": False,
            "square_cost": False,
            "n_units": 64,
            "cost_param": 0.0002 / 2 / 2 / 10 / 100 / 10 / 5,
            "weight_decay_pl_mul": 1.0 /100 / 100 / 100 / 10,
            "planner_std_theta_mul": 1.0,

            # "lr": tune.grid_search([0.01/3.0, 0.01/3.0/3.0, 0.01/3.0/3.0/3.0, 0.01/3.0/3.0/3.0/3.0]),
            # "momentum": tune.grid_search([0.99,0.999]),
            # "planner_momentum": tune.grid_search([0.99,0.999]),
            # "cost_param": tune.grid_search([0.0002 / 2 / 2 / 10 / 100 / 10 / 5,
            #                                 0.0002 / 2 / 2 / 10 / 100 / 10 / 5*3,
            #                                 0.0002 / 2 / 2 / 10 / 100 / 10 / 5/3]),

            "add_state_grad": True,
            "normalize_against_v": 1000,
            # "loss_mul_planner": 100.0 * 100 * 10 / 3 / 5 / 10 * 3,
            # "planner_std_theta_mul": 0.5,
            # "cost_param": 0.0002 / 2 / 2 / 10 / 100 / 10 / 5 * 3,
            "planner_momentum": 0.999,
            "entropy_coeff": 0.1 / 10 / 3 * 3 * 3,

            "loss_mul_planner": 100.0 * 100 * 10 / 3 / 5 / 10 * 3,
            "planner_std_theta_mul": 1.0,
            "cost_param": 0.0002 / 2 / 2 / 10 / 100 / 10 / 5 * 3 / 3 / 3,
            "weight_decay_pl_mul": 1.0 /100 / 100 / 100 / 10 * 3 / 10,
            # "cost_param": tune.grid_search([0.0002 / 2 / 2 / 10 / 100 / 10 / 5 * 3,
            #                                 0.0002 / 2 / 2 / 10 / 100 / 10 / 5 * 3 / 3,
            #                                 0.0002 / 2 / 2 / 10 / 100 / 10 / 5 * 3 / 3 / 3,
            #                                 0.0002 / 2 / 2 / 10 / 100 / 10 / 5 * 3 * 3,
            #                                 0.0002 / 2 / 2 / 10 / 100 / 10 / 5 * 3 * 3 * 3]),
            # "loss_mul_planner": tune.grid_search([
            #     100.0 * 100 * 10 / 3 / 5 / 10 * 3 * 3,
            #     100.0 * 100 * 10 / 3 / 5 / 10 * 3,
            #     100.0 * 100 * 10 / 3 / 5 / 10 * 3 / 3]),
            # "weight_decay_pl_mul": tune.grid_search([
            #     1.0 /100 / 100 / 100 / 10 / 3,
            #     1.0 /100 / 100 / 100 / 10,
            #     1.0 /100 / 100 / 100 / 10 * 3]),
            "normalize_against_vp": 1000,

            "add_state_grad": True,
            "loss_mul_planner": 2000,
            "cost_param": 2.7e-7,
            "weight_decay_pl_mul": 3e-6,

            # "with_planner": tune.grid_search([True, False]),

            # "weight_decay_pl_mul": tune.grid_search([3e-6, 1e-6]),
            # "cost_param": tune.grid_search([2.7e-7, 1e-8]),
            # "loss_mul_planner": tune.grid_search([2000, 6000]),
            # "entropy_coeff": tune.grid_search([
            #     0.1 / 10 / 3 * 3 * 3,
            #     0.1 / 10 / 3 * 3 * 3 * 3, ]),
            # "normalize_vp_separated": True,
            # "add_state_grad": tune.grid_search([True, False]),

            "planner_clip_norm": 3e-6,
            "loss_mul_planner": 2000,
            "cost_param": 3e-8,
            "weight_decay_pl_mul": 1e-7,
            "entropy_coeff": 0.1,
            "normalize_vp_separated": True,

            # "weight_decay_pl_mul": tune.grid_search([3e-7, 1e-7]),
            # "cost_param": tune.grid_search([1e-8, 3e-9]),
            # "planner_clip_norm": tune.grid_search([3e-6, 1e-6]),
            # "add_state_grad": tune.grid_search([True, False]),
            # "normalize_vp_separated": tune.grid_search([True, False]),

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
        "report_every_n": 1,

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

        # "with_planner": True,
        "with_planner": False,
        # "with_planner": tune.grid_search([True, False]),

        # "env": "FearGreedMatrix",
        "env": "CoinGame",

        "normalize_against_vp": False,
        "normalize_against_v": False,
        "normalize_vp_separated": False,

    }

    hyperparameters = add_env_hp(hyperparameters)

    train(hyperparameters)


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
