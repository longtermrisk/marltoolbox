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

from marltoolbox.algos.sos import SOSTrainer
from marltoolbox.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
    IteratedMatchingPennies,
    IteratedAsymBoS,
)
from marltoolbox.experiments.tune_class_api import lola_exact_meta_game
from marltoolbox.experiments.tune_class_api import lola_pg_official
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import policy, log, miscellaneous


def main(debug):
    hparams = get_hyperparameters(debug)

    if hparams["load_plot_data"] is None:
        ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=0,
            local_mode=debug,
        )
        tune_analysis_per_exp = train(hparams)
    else:
        tune_analysis_per_exp = None

    evaluate(tune_analysis_per_exp, hparams)
    ray.shutdown()


def get_hyperparameters(debug, train_n_replicates=None, env=None):
    """Get hyperparameters for LOLA-Exact for matrix games"""

    if train_n_replicates is None:
        train_n_replicates = 2 if debug else int(3 * 2)
    seeds = miscellaneous.get_random_seeds(train_n_replicates)

    exp_name, _ = log.log_in_current_day_dir("SOS")

    hparams = {
        "debug": debug,
        "load_plot_data": None,
        "exp_name": exp_name,
        "classify_into_welfare_fn": True,
        "train_n_replicates": train_n_replicates,
        "wandb": {
            "project": "SOS",
            "group": exp_name,
            "api_key_file": os.path.join(
                os.path.dirname(__file__), "../../../api_key_wandb"
            ),
        },
        "env_name": "IteratedAsymBoS" if env is None else env,
        "lr": 1.0 / 10,
        "gamma": 0.96,
        "num_epochs": 5 if debug else 100,
        # "method": "lola",
        "method": "sos",
        "inital_weights_std": 1.0,
        "seed": tune.grid_search(seeds),
        "metric": "mean_reward_player_row",
        "plot_keys": aggregate_and_plot_tensorboard_data.PLOT_KEYS
        + ["mean_reward"],
        "plot_assemblage_tags": aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
        + [("mean_reward",)],
        "x_limits": (-0.1, 4.1),
        "y_limits": (-0.1, 4.1),
        "max_steps_in_eval": 100,
    }

    return hparams


def train(hp):
    tune_config, stop_config, _ = get_tune_config(hp)
    # Train with the Tune Class API (not an RLLib Trainer)
    tune_analysis = tune.run(
        SOSTrainer,
        name=hp["exp_name"],
        config=tune_config,
        checkpoint_at_end=True,
        stop=stop_config,
        metric=hp["metric"],
        mode="max",
    )
    if hp["classify_into_welfare_fn"]:
        tune_analysis_per_exp = _split_tune_results_wt_welfare(tune_analysis)
    else:
        tune_analysis_per_exp = {"": tune_analysis}

    aggregate_and_plot_tensorboard_data.add_summary_plots(
        main_path=os.path.join("~/ray_results/", tune_config["exp_name"]),
        plot_keys=tune_config["plot_keys"],
        plot_assemble_tags_in_one_plot=tune_config["plot_assemblage_tags"],
    )

    return tune_analysis_per_exp


def get_tune_config(hp: dict):
    tune_config = copy.deepcopy(hp)
    assert tune_config["env_name"] in ("IPD", "IteratedAsymBoS")
    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": hp["max_steps_in_eval"],
        "get_additional_info": True,
    }
    tune_config["plot_axis_scale_multipliers"] = (
        (
            1 / hp["max_steps_in_eval"],
            1 / hp["max_steps_in_eval"],
        ),
    )
    if "num_episodes" in tune_config:
        stop_config = {"episodes_total": tune_config["num_episodes"]}
    else:
        stop_config = {"episodes_total": tune_config["num_epochs"]}

    return tune_config, stop_config, env_config


def evaluate(tune_analysis_per_exp, hp):
    (
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    ) = generate_eval_config(hp)

    lola_pg_official._evaluate_self_and_cross_perf(
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
        tune_analysis_per_exp,
        n_cross_play_per_checkpoint=min(15, hp["train_n_replicates"] - 1)
        if hp["classify_into_welfare_fn"]
        else None,
    )


def generate_eval_config(hp):
    hp_eval = copy.deepcopy(hp)

    hp_eval["min_iter_time_s"] = 3.0
    hp_eval["seed"] = miscellaneous.get_random_seeds(1)[0]
    hp_eval["batch_size"] = 1
    hp_eval["num_episodes"] = 100

    tune_config, stop_config, env_config = get_tune_config(hp_eval)
    tune_config["TuneTrainerClass"] = SOSTrainer

    hp_eval["group_names"] = ["lola"]
    hp_eval["scale_multipliers"] = (
        1 / hp_eval["max_steps_in_eval"],
        1 / hp_eval["max_steps_in_eval"],
    )
    hp_eval["jitter"] = 0.05

    if hp_eval["env_name"] == "IPD":
        hp_eval["env_class"] = IteratedPrisonersDilemma
        hp_eval["x_limits"] = (-3.5, 0.5)
        hp_eval["y_limits"] = (-3.5, 0.5)
    elif hp_eval["env_name"] == "IMP":
        hp_eval["env_class"] = IteratedMatchingPennies
        hp_eval["x_limits"] = (-1.0, 1.0)
        hp_eval["y_limits"] = (-1.0, 1.0)
    elif hp_eval["env_name"] == "IteratedAsymBoS":
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
                    {"tune_config": copy.deepcopy(tune_config)},
                ),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(PGTorchPolicy),
                    hp_eval["env_class"](env_config).OBSERVATION_SPACE,
                    hp_eval["env_class"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)},
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": hp_eval["seed"],
        "min_iter_time_s": hp_eval["min_iter_time_s"],
        "num_workers": 0,
        "num_envs_per_worker": 1,
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])
    trainable_class = SOSTrainer

    return (
        hp_eval,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    )


def _split_tune_results_wt_welfare(
    tune_analysis,
):
    tune_analysis_per_welfare = {}
    for trial in tune_analysis.trials:
        welfare_name = _get_trial_welfare(trial)
        if welfare_name not in tune_analysis_per_welfare.keys():
            _add_empty_tune_analysis(
                tune_analysis_per_welfare, welfare_name, tune_analysis
            )
        tune_analysis_per_welfare[welfare_name].trials.append(trial)
    return tune_analysis_per_welfare


def _get_trial_welfare(trial):
    reward_player_1 = trial.last_result["mean_reward_player_row"]
    reward_player_2 = trial.last_result["mean_reward_player_col"]
    welfare_name = lola_exact_meta_game.classify_into_welfare_based_on_rewards(
        reward_player_1, reward_player_2
    )
    return welfare_name


def _add_empty_tune_analysis(
    tune_analysis_per_welfare, welfare_name, tune_analysis
):
    tune_analysis_per_welfare[welfare_name] = copy.deepcopy(tune_analysis)
    tune_analysis_per_welfare[welfare_name].trials = []


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
