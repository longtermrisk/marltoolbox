import copy
import logging
import os
import time

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTorchPolicy
from ray.tune.integration.wandb import WandbLoggerCallback

from marltoolbox.algos.lola import (
    train_cg_tune_class_API,
    train_pg_tune_class_API,
)
from marltoolbox.envs import (
    simple_bargaining,
)
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import policy, log, cross_play, exp_analysis
from marltoolbox.utils.plot import PlotConfig
from marltoolbox.experiments.tune_class_api import lola_exact_official
from marltoolbox.algos.psro import PSROTrainer

logger = logging.getLogger(__name__)


def main(debug: bool, env=None):
    """
    Train several LOLA_PG pairs of agent on the selected environment and
    plot their performances in self-play and cross-play.

    :param debug: selection of debug mode using less compute
    :param env: option to overwrite the env selection
    """
    train_n_replicates = 2 if debug else 40
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("PSRO")

    tune_hparams = _get_hyperparameters(
        debug, train_n_replicates, seeds, exp_name, env
    )

    if tune_hparams["load_plot_data"] is None:
        ray.init(num_cpus=10, num_gpus=0, local_mode=debug)
        experiment_analysis_per_welfare = _train(tune_hparams)
    else:
        experiment_analysis_per_welfare = None

    _evaluate(tune_hparams, debug, experiment_analysis_per_welfare)
    ray.shutdown()


def _get_hyperparameters(debug, train_n_replicates, seeds, exp_name, env):

    tune_hparams = {
        "debug": debug,
        "exp_name": exp_name,
        "train_n_replicates": train_n_replicates,
        "wandb": {
            "project": "LOLA_PG",
            "group": exp_name,
            "api_key_file": os.path.join(
                os.path.dirname(__file__), "../../../api_key_wandb"
            ),
        },
        # "classify_into_welfare_fn": True,
        "seed": tune.grid_search(seeds),
        "load_plot_data": None,
        # Example: "load_plot_data": ".../SelfAndCrossPlay_save.p",
        "n_players": 2,
        # "game_name": "kuhn_poker",
        "game_name": "python_simple_bargaining",
        "oracle_type": "BR",
        "training_strategy_selector": "probabilistic",
        "rectifier": "",
        "sims_per_entry": 1000,
        "number_policies_selected": 1,
        "meta_strategy_method": "alpharank",
        "symmetric_game": False,
        "verbose": True,
        "loss_str": "qpg",
        "hidden_layer_size": 256,
        "n_hidden_layers": 4,
        "batch_size": 32,
        "entropy_cost": 0.001,
        "critic_learning_rate": 1e-2,
        "pi_learning_rate": 1e-3,
        "num_q_before_pi": 8,
        "optimizer_str": "adam",
        "self_play_proportion": 0.0,
        "number_training_episodes": int(1e4),
        "play_proportion": 0.0,
        "sigma": 0.0,
        "dqn_learning_rate": 1e-2,
        "update_target_network_every": 1000,
        "learn_every": 10,
        "num_iterations": 100,
        "plot_keys": [
            "reward",
            "total_reward",
            "entrop",
        ],
        "plot_assemblage_tags": [
            ("total_reward",),
            ("entrop",),
        ],
    }
    return tune_hparams


def _train(tune_hp):
    tune_config, stop, env_config = _get_tune_config(tune_hp)

    # Train with the Tune Class API (not RLLib Class)
    experiment_analysis = tune.run(
        PSROTrainer,
        name=tune_hp["exp_name"],
        config=tune_config,
        checkpoint_at_end=True,
        stop=stop,
        metric=tune_config["metric"],
        mode="max",
        log_to_file=True,  # not tune_hp["debug"],
        callbacks=None
        if tune_hp["debug"]
        else [
            WandbLoggerCallback(
                project=tune_hp["wandb"]["project"],
                group=tune_hp["wandb"]["group"],
                api_key_file=tune_hp["wandb"]["api_key_file"],
                log_config=True,
            )
        ],
    )

    if tune_hp["classify_into_welfare_fn"]:
        experiment_analysis_per_welfare = (
            _classify_trials_in_function_of_welfare(
                experiment_analysis, tune_hp
            )
        )
    else:
        experiment_analysis_per_welfare = {"": experiment_analysis}

    aggregate_and_plot_tensorboard_data.add_summary_plots(
        main_path=os.path.join("~/ray_results/", tune_config["exp_name"]),
        plot_keys=tune_config["plot_keys"],
        plot_assemble_tags_in_one_plot=tune_config["plot_assemblage_tags"],
    )

    return experiment_analysis_per_welfare


def _get_tune_config(tune_hp: dict, stop_on_epi_number: bool = False):
    tune_config = copy.deepcopy(tune_hp)

    tune_config["plot_keys"] += (
        train_cg_tune_class_API.PLOT_KEYS
        + aggregate_and_plot_tensorboard_data.PLOT_KEYS
    )
    tune_config["plot_assemblage_tags"] += (
        train_cg_tune_class_API.PLOT_ASSEMBLAGE_TAGS
        + aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
    )
    tune_hp["x_limits"] = (-3.0, 3.0)
    tune_hp["y_limits"] = (-3.0, 3.0)
    tune_hp["jitter"] = 0.05
    tune_config["metric"] = "training_iteration"
    env_config = {}
    # For hyperparameter search
    tune_hp["scale_multipliers"] = tune.sample_from(
        lambda spec: (
            1 / 1,
            1 / 1,
        )
    )
    if stop_on_epi_number:
        stop = {"episodes_total": tune_config["num_iterations"]}
    else:
        stop = {"finished": True}

    return tune_config, stop, env_config


def _evaluate(tune_hp, debug, experiment_analysis_per_exp):
    (
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop,
        env_config,
    ) = _generate_eval_config(tune_hp, debug)

    _evaluate_self_and_cross_perf(
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop,
        env_config,
        experiment_analysis_per_exp,
    )


def _generate_eval_config(tune_hp, debug):
    rllib_hp = copy.deepcopy(tune_hp)
    rllib_hp["seed"] = 2020
    rllib_hp["num_episodes"] = 1 if debug else 100
    tune_config, stop, env_config = _get_tune_config(
        rllib_hp, stop_on_epi_number=True
    )
    rllib_hp["env_class"] = tune_config["env_class"]

    if "CoinGame" in tune_config["env_name"]:
        env_config["batch_size"] = 1
        tune_config["TuneTrainerClass"] = train_cg_tune_class_API.LOLAPGCG
    else:
        tune_config["TuneTrainerClass"] = train_pg_tune_class_API.LOLAPGMatrice
    tune_config["env_config"].update(
        {
            "batch_size": env_config["batch_size"],
            "max_steps": rllib_hp["trace_length"],
        }
    )
    rllib_hp["scale_multipliers"] = (
        1 / rllib_hp["trace_length"],
        1 / rllib_hp["trace_length"],
    )

    rllib_config_eval = {
        "env": rllib_hp["env_class"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    # The default policy is DQN defined in DQNTrainer
                    # but we overwrite it to use the LE policy
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env_class"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env_class"].ACTION_SPACE,
                    {"tune_config": tune_config},
                ),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env_class"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env_class"].ACTION_SPACE,
                    {"tune_config": tune_config},
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": rllib_hp["seed"],
        "min_iter_time_s": 3.0,
        "callbacks": log.get_logging_callbacks_class(
            log_full_epi=True,
        ),
        "num_envs_per_worker": 1,
        "num_workers": 0,
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])

    if "CoinGame" in rllib_hp["env_name"]:
        trainable_class = train_cg_tune_class_API.LOLAPGCG
        rllib_config_eval["model"] = {
            "dim": env_config["grid_size"],
            # [Channel, [Kernel, Kernel], Stride]]
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
        }
    else:
        trainable_class = train_pg_tune_class_API.LOLAPGMatrice

    return (
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop,
        env_config,
    )


def _evaluate_self_and_cross_perf(
    rllib_hp,
    rllib_config_eval,
    policies_to_load,
    trainable_class,
    stop,
    env_config,
    experiment_analysis_per_welfare,
    n_cross_play_per_checkpoint=None,
):
    exp_name = os.path.join(rllib_hp["exp_name"], "eval")
    evaluator = cross_play.evaluator.SelfAndCrossPlayEvaluator(
        exp_name=exp_name,
        local_mode=rllib_hp["debug"],
    )
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=rllib_config_eval,
        stop_config=stop,
        policies_to_load_from_checkpoint=policies_to_load,
        experiment_analysis_per_welfare=experiment_analysis_per_welfare,
        tune_trainer_class=trainable_class,
        n_cross_play_per_checkpoint=min(5, rllib_hp["train_n_replicates"] - 1)
        if n_cross_play_per_checkpoint is None
        else n_cross_play_per_checkpoint,
        to_load_path=rllib_hp["load_plot_data"],
    )

    plot_config = PlotConfig(
        xlim=rllib_hp["x_limits"],
        ylim=rllib_hp["y_limits"],
        markersize=5,
        jitter=rllib_hp["jitter"],
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        plot_max_n_points=rllib_hp["train_n_replicates"],
        x_scale_multiplier=rllib_hp["scale_multipliers"][0],
        y_scale_multiplier=rllib_hp["scale_multipliers"][1],
    )
    evaluator.plot_results(
        analysis_metrics_per_mode,
        plot_config=plot_config,
        x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}",
    )


FAILURE = "failures"
EGALITARIAN = "egalitarian"
UTILITARIAN = "utilitarian"


def _classify_trials_in_function_of_welfare(experiment_analysis, hp):
    experiment_analysis_per_welfare = {}
    for trial in experiment_analysis.trials:
        welfare_name = _get_trial_welfare(trial, hp)
        if welfare_name not in experiment_analysis_per_welfare.keys():
            lola_exact_official._add_empty_experiment_analysis(
                experiment_analysis_per_welfare,
                welfare_name,
                experiment_analysis,
            )
        experiment_analysis_per_welfare[welfare_name].trials.append(trial)
    return experiment_analysis_per_welfare


def _get_trial_welfare(trial, hp):
    pick_own_player_1 = trial.last_result["player_red_pick_own_color"]
    pick_own_player_2 = trial.last_result["player_blue_pick_own_color"]
    reward_player_1 = trial.last_result["total_reward_player_red"]
    reward_player_2 = trial.last_result["total_reward_player_blue"]
    welfare_name = lola_pg_classify_fn(
        pick_own_player_1,
        pick_own_player_2,
        hp,
        reward_player_1,
        reward_player_2,
    )
    return welfare_name


def lola_pg_classify_fn(
    pick_own_player_1, pick_own_player_2, hp, reward_player_1, reward_player_2
):
    if reward_player_2 != 0.0 and reward_player_1 != 0.0:
        if hp["env_name"] == "VectorizedSSDMixedMotiveCoinGame":
            ratio = reward_player_2 / reward_player_1
        else:
            ratio = max(
                reward_player_1 / reward_player_2,
                reward_player_2 / reward_player_1,
            )
        if ratio > 1.2:
            return UTILITARIAN
    return EGALITARIAN


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
