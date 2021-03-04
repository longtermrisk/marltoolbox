##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the tune_class_api/lola_pg_official.py file
##########

import copy
import os
import time

import ray
from ray import tune
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy

from marltoolbox.algos.lola_dice.train_tune_class_API import LOLADICE
from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame
from marltoolbox.envs.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma, IteratedMatchingPennies, IteratedAsymBoS
from marltoolbox.examples.tune_class_api import lola_pg_official
from marltoolbox.utils import policy, log


def train(hp):
    tune_config, stop, _ = get_tune_config(hp)
    # Train with the Tune Class API (not RLLib Class)
    tune_analysis = tune.run(LOLADICE, name=hp["exp_name"], config=tune_config,
                             checkpoint_at_end=True,
                             stop=stop, metric=hp["metric"], mode="max")
    tune_analysis_per_exp = {"": tune_analysis}
    return tune_analysis_per_exp


def get_tune_config(hp: dict) -> dict:
    config = copy.deepcopy(hp)
    assert config['env'] in ("CoinGame", "IPD", "IMP", "AsymCoinGame", "BoS", "AsymBoS")

    if config["env"] in ("IPD", "IMP", "BoS", "AsymBoS"):
        config["make_policy"] = ("make_simple_policy", {})
        config["base_lr"] = 1.0

        config["trace_length"] = 150 if config["trace_length"] is None else config["trace_length"]
        config["make_optimizer"] = ("make_sgd_optimizer", {})

        env_config = {
            "players_ids": ["player_row", "player_col"],
            "batch_size": config["batch_size"],
            "max_steps": config["trace_length"],
            "get_additional_info": True,
        }

    if config["env"] in ("IPD", "BoS", "AsymBoS"):
        config["gamma"] = 0.96 if config["gamma"] is None else config["gamma"]
        config["save_dir"] = "dice_results_ipd"
    elif config["env"] == "IMP":
        config["gamma"] = 0.9 if config["gamma"] is None else config["gamma"]
        config["save_dir"] = "dice_results_imp"
    elif config["env"] in ("CoinGame", "AsymCoinGame"):
        config["trace_length"] = 150 if config["trace_length"] is None else config["trace_length"]
        config["epochs"] *= 10
        config["make_optimizer"] = ("make_adam_optimizer", {"hidden_sizes": [16, 32]})
        config["save_dir"] = "dice_results_coin_game"
        config["gamma"] = 0.96 if config["gamma"] is None else config["gamma"]
        config["make_policy"] = ("make_conv_policy", {})
        config["base_lr"] = 0.005

        env_config = {
            "players_ids": ["player_row", "player_col"],
            "batch_size": config["batch_size"],
            "max_steps": config["trace_length"],
            "get_additional_info": True,
            "grid_size": config["grid_size"],
        }

    config["lr_inner"] = config["lr_inner"] * config["base_lr"]
    config["lr_outer"] = config["lr_outer"] * config["base_lr"]
    config["lr_value"] = config["lr_value"] * config["base_lr"]
    config["lr_om"] = config["lr_om"] * config["base_lr"]

    stop = {"episodes_total": config['epochs']}

    return config, stop, env_config


def evaluate(tune_analysis_per_exp, hp, debug):
    (rllib_hp, rllib_config_eval, policies_to_load, trainable_class, stop, env_config) = \
        generate_eval_config(hp, debug)

    lola_pg_official.evaluate_self_and_cross_perf(rllib_hp, rllib_config_eval, policies_to_load,
                                                  trainable_class, stop, env_config, tune_analysis_per_exp)


def generate_eval_config(hp, debug):
    hp_eval = copy.deepcopy(hp)

    hp_eval["min_iter_time_s"] = 3.0
    hp_eval["seed"] = 2020
    hp_eval["batch_size"] = 1
    hp_eval["num_episodes"] = 3 if debug else 100
    tune_config, stop, env_config = get_tune_config(hp_eval)
    tune_config['TuneTrainerClass'] = LOLADICE

    hp_eval["group_names"] = ["lola"]
    hp_eval["scale_multipliers"] = (1 / tune_config['trace_length'], 1 / tune_config['trace_length'])
    hp_eval["jitter"] = 0.05

    if hp_eval["env"] == "IPD":
        hp_eval["env"] = IteratedPrisonersDilemma
        hp_eval["x_limits"] = (-3.5, 0.5)
        hp_eval["y_limits"] = (-3.5, 0.5)
    elif hp_eval["env"] == "IMP":
        hp_eval["env"] = IteratedMatchingPennies
        hp_eval["x_limits"] = (-1.0, 1.0)
        hp_eval["y_limits"] = (-1.0, 1.0)
    elif hp_eval["env"] == "AsymBoS":
        hp_eval["env"] = IteratedAsymBoS
        hp_eval["x_limits"] = (-0.5, 4.0)
        hp_eval["y_limits"] = (-0.5, 4.0)
    elif hp_eval["env"] == "CoinGame":
        hp_eval["env"] = CoinGame
        hp_eval["x_limits"] = (-1.0, 3.0)
        hp_eval["y_limits"] = (-1.0, 3.0)
        hp_eval["jitter"] = 0.02
        env_config["force_vectorize"] = False
    elif hp_eval["env"] == "AsymCoinGame":
        hp_eval["env"] = AsymCoinGame
        hp_eval["x_limits"] = (-1.0, 3.0)
        hp_eval["y_limits"] = (-1.0, 3.0)
        hp_eval["jitter"] = 0.02
        env_config["force_vectorize"] = False
    else:
        raise NotImplementedError()

    rllib_config_eval = {
        "env": hp_eval["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    hp_eval["env"](env_config).OBSERVATION_SPACE,
                    hp_eval["env"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)}),

                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    hp_eval["env"](env_config).OBSERVATION_SPACE,
                    hp_eval["env"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": hp_eval["seed"],
        "min_iter_time_s": hp_eval["min_iter_time_s"],
    }
    policies_to_load = copy.deepcopy(env_config["players_ids"])
    trainable_class = tune_config['TuneTrainerClass']

    return hp_eval, rllib_config_eval, policies_to_load, trainable_class, stop, env_config


def main(debug):
    train_n_replicates = 2 if debug else 40
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("LOLA_DICE")

    hparams = {

        "load_plot_data": None,
        # IPD
        # Example: "load_plot_data": ".../SameAndCrossPlay_save.p",

        "exp_name": exp_name,
        "train_n_replicates": train_n_replicates,
        "env": "IPD",
        # "env": "IMP",
        # "env": "AsymBoS",
        # "env": "CoinGame",
        # "env": "AsymCoinGame",

        "gamma": None,
        "trace_length": 10 if debug else None,

        "epochs": 2 if debug else 200,
        "lr_inner": .1,
        "lr_outer": .2,
        "lr_value": .1,
        "lr_om": .1,
        "inner_asymm": True,
        "n_agents": 2,
        "n_inner_steps": 1 if debug else 2,
        "batch_size": 10 if debug else 64,
        "value_batch_size": 16,
        "value_epochs": 0,
        "om_batch_size": 16,
        "om_epochs": 0,
        "grid_size": 3,
        "use_baseline": False,
        "use_dice": True,
        "use_opp_modeling": False,

        "seed": tune.grid_search(seeds),
        "metric": "ag_0_returns_player_1",
    }

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)
        training_results = train(hparams)
    else:
        training_results = None

    evaluate(training_results, hparams, debug)
    ray.shutdown()


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
