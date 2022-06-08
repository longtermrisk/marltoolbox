##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the
# tune_class_api/lola_pg_official.py file
##########

import copy
import os

import ray
from ray import tune
from ray.tune.analysis import ExperimentAnalysis
from ray.rllib.agents.pg import PGTorchPolicy
from ray.tune.integration.wandb import WandbLoggerCallback
from marltoolbox.experiments.tune_class_api import lola_exact_meta_game

from marltoolbox.algos.lola.train_exact_tune_class_API import LOLAExactTrainer
from marltoolbox.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
    IteratedMatchingPennies,
    IteratedAsymBoS,
)
from marltoolbox.experiments.tune_class_api import lola_pg_official
from marltoolbox.utils import policy, log, miscellaneous
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.algos.lola import utils


def main(debug):
    hparams = utils.get_hyperparameters(debug)

    if hparams["load_plot_data"] is None:
        ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=0,
            local_mode=debug,
        )
        experiment_analysis_per_welfare = train(hparams)
    else:
        experiment_analysis_per_welfare = None

    evaluate(experiment_analysis_per_welfare, hparams)
    ray.shutdown()


def train(hp):
    tune_config, stop_config, _ = get_tune_config(hp)
    # Train with the Tune Class API (not an RLLib Trainer)
    experiment_analysis = tune.run(
        LOLAExactTrainer,
        name=hp["exp_name"],
        config=tune_config,
        checkpoint_at_end=True,
        stop=stop_config,
        metric=hp["metric"],
        mode="max",
        # callbacks=None
        # if hp["debug"]
        # else [
        #     WandbLoggerCallback(
        #         project=hp["wandb"]["project"],
        #         group=hp["wandb"]["group"],
        #         api_key_file=hp["wandb"]["api_key_file"],
        #         log_config=True,
        #     )
        # ],
    )
    if hp["classify_into_welfare_fn"]:
        experiment_analysis_per_welfare = _classify_trials_in_function_of_welfare(
            experiment_analysis
        )
    else:
        experiment_analysis_per_welfare = {"": experiment_analysis}

    return experiment_analysis_per_welfare


def get_tune_config(hp: dict):
    tune_config = copy.deepcopy(hp)
    assert tune_config["env_name"] in ("IPD", "IMP", "BoS", "IteratedAsymBoS")

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": tune_config["trace_length"],
        "get_additional_info": True,
    }

    if tune_config["env_name"] == "IteratedAsymBoS":
        tune_config["Q_net_std"] = 3.0
    else:
        tune_config["Q_net_std"] = 1.0

    if tune_config["env_name"] in ("IPD", "BoS", "IteratedAsymBoS"):
        tune_config["gamma"] = (
            0.96 if tune_config["gamma"] is None else tune_config["gamma"]
        )
        tune_config["save_dir"] = "dice_results_ipd"
    elif tune_config["env_name"] == "IMP":
        tune_config["gamma"] = (
            0.9 if tune_config["gamma"] is None else tune_config["gamma"]
        )
        tune_config["save_dir"] = "dice_results_imp"

    stop_config = {"episodes_total": tune_config["num_episodes"]}
    return tune_config, stop_config, env_config


def evaluate(experiment_analysis_per_welfare, hp):
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
        experiment_analysis_per_welfare,
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
    tune_config["TuneTrainerClass"] = LOLAExactTrainer

    hp_eval["group_names"] = ["lola"]
    hp_eval["scale_multipliers"] = (
        1 / tune_config["trace_length"],
        1 / tune_config["trace_length"],
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
    trainable_class = LOLAExactTrainer

    return (
        hp_eval,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    )


def _classify_trials_in_function_of_welfare(
    experiment_analysis,
):
    experiment_analysis_per_welfare = {}
    for trial in experiment_analysis.trials:
        welfare_name = _get_trial_welfare(trial)
        if welfare_name not in experiment_analysis_per_welfare.keys():
            _add_empty_experiment_analysis(
                experiment_analysis_per_welfare,
                welfare_name,
                experiment_analysis,
            )
        experiment_analysis_per_welfare[welfare_name].trials.append(trial)
    return experiment_analysis_per_welfare


def _get_trial_welfare(trial):
    reward_player_1 = trial.last_result["ret1"]
    reward_player_2 = trial.last_result["ret2"]
    welfare_name = lola_exact_meta_game.classify_into_welfare_based_on_rewards(
        reward_player_1, reward_player_2
    )
    return welfare_name


def _add_empty_experiment_analysis(
    experiment_analysis_per_welfare, welfare_name, experiment_analysis
):
    experiment_analysis_per_welfare[welfare_name] = copy.deepcopy(experiment_analysis)
    experiment_analysis_per_welfare[welfare_name].trials = []


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
