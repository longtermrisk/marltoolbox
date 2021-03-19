##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the
# tune_class_api/lola_pg_official.py file
##########

import copy
import os

import ray
from ray import tune
from ray.rllib.agents.pg import PGTorchPolicy

from marltoolbox.algos.lola.train_exact_tune_class_API import LOLAExact
from marltoolbox.envs.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma, IteratedMatchingPennies, IteratedAsymBoS
from marltoolbox.examples.tune_class_api import lola_pg_official
from marltoolbox.utils import policy, log, miscellaneous


def main(debug):
    train_n_replicates = 2 if debug else 40
    seeds = miscellaneous.get_random_seeds(train_n_replicates)

    exp_name, _ = log.log_in_current_day_dir("LOLA_Exact")

    hparams = {

        "load_plot_data": None,
        # Example "load_plot_data": ".../SelfAndCrossPlay_save.p",

        "exp_name": exp_name,
        "train_n_replicates": train_n_replicates,

        "env_name": "IPD",
        # "env_name": "IMP",
        # "env_name": "AsymBoS",

        "num_episodes": 5 if debug else 50,
        "trace_length": 5 if debug else 200,
        "simple_net": True,
        "corrections": True,
        "pseudo": False,
        "num_hidden": 32,
        "reg": 0.0,
        "lr": 1.,
        "lr_correction": 1.0,
        "gamma": 0.96,

        "seed": tune.grid_search(seeds),
        "metric": "ret1",

        "with_linear_LR_decay_to_zero": False,
        "clip_update": None,

        # "with_linear_LR_decay_to_zero": True,
        # "clip_update": 0.1,
        # "lr": 0.001,

    }

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)
        tune_analysis_per_exp = train(hparams)
    else:
        tune_analysis_per_exp = None

    evaluate(tune_analysis_per_exp, hparams)
    ray.shutdown()


def train(hp):
    tune_config, stop, _ = get_tune_config(hp)
    # Train with the Tune Class API (not an RLLib Trainer)
    tune_analysis = tune.run(LOLAExact, name=hp["exp_name"], config=tune_config,
                             checkpoint_at_end=True,
                             stop=stop, metric=hp["metric"], mode="max")
    tune_analysis_per_exp = {"": tune_analysis}
    return tune_analysis_per_exp


def get_tune_config(hp: dict) -> dict:
    tune_config = copy.deepcopy(hp)
    assert tune_config['env_name'] in ("IPD", "IMP", "BoS", "AsymBoS")

    if tune_config["env_name"] in ("IPD", "IMP", "BoS", "AsymBoS"):
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": tune_config["trace_length"],
            "get_additional_info": True,
        }

    if tune_config["env_name"] in ("IPD", "BoS", "AsymBoS"):
        tune_config["gamma"] = 0.96 \
            if tune_config["gamma"] is None \
            else tune_config["gamma"]
        tune_config["save_dir"] = "dice_results_ipd"
    elif tune_config["env_name"] == "IMP":
        tune_config["gamma"] = 0.9 \
            if tune_config["gamma"] is None \
            else tune_config["gamma"]
        tune_config["save_dir"] = "dice_results_imp"

    stop = {"episodes_total": tune_config['num_episodes']}
    return tune_config, stop, env_config


def evaluate(tune_analysis_per_exp, hp):
    (rllib_hp, rllib_config_eval, policies_to_load,
     trainable_class, stop, env_config) = generate_eval_config(hp)

    lola_pg_official.evaluate_self_and_cross_perf(
        rllib_hp, rllib_config_eval, policies_to_load,
        trainable_class, stop, env_config, tune_analysis_per_exp)


def generate_eval_config(hp):
    hp_eval = copy.deepcopy(hp)

    hp_eval["min_iter_time_s"] = 3.0
    hp_eval["seed"] = miscellaneous.get_random_seeds(1)[0]
    hp_eval["batch_size"] = 1
    hp_eval["num_episodes"] = 100

    tune_config, stop, env_config = get_tune_config(hp_eval)
    tune_config['TuneTrainerClass'] = LOLAExact

    hp_eval["group_names"] = ["lola"]
    hp_eval["scale_multipliers"] = (1 / tune_config['trace_length'],
                                    1 / tune_config['trace_length'])
    hp_eval["jitter"] = 0.05

    if hp_eval["env_name"] == "IPD":
        hp_eval["env_class"] = IteratedPrisonersDilemma
        hp_eval["x_limits"] = (-3.5, 0.5)
        hp_eval["y_limits"] = (-3.5, 0.5)
    elif hp_eval["env_name"] == "IMP":
        hp_eval["env_class"] = IteratedMatchingPennies
        hp_eval["x_limits"] = (-1.0, 1.0)
        hp_eval["y_limits"] = (-1.0, 1.0)
    elif hp_eval["env_name"] == "AsymBoS":
        hp_eval["env_class"] = IteratedAsymBoS
        hp_eval["x_limits"] = (-0.1, 4.1)
        hp_eval["y_limits"] = (-0.1, 4.1)
    else:
        raise NotImplementedError()

    rllib_config_eval = {
        "env": hp_eval["env_class"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    policy.get_tune_policy_class(PGTorchPolicy),
                    hp_eval["env_class"](env_config).OBSERVATION_SPACE,
                    hp_eval["env_class"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)}),

                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(PGTorchPolicy),
                    hp_eval["env_class"](env_config).OBSERVATION_SPACE,
                    hp_eval["env_class"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": hp_eval["seed"],
        "min_iter_time_s": hp_eval["min_iter_time_s"],
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])
    trainable_class = LOLAExact

    return hp_eval, rllib_config_eval, policies_to_load, \
           trainable_class, stop, env_config


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
