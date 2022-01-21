import copy
import logging
import os
import argparse

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.exploration import StochasticSampling
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS

from marltoolbox.algos import amTFT, augmented_r2d2
from marltoolbox.algos.augmented_ppo import MyPPOTorchPolicy
from marltoolbox.envs import (
    matrix_sequential_social_dilemma,
    coin_game,
    ssd_mixed_motive_coin_game,
)
from marltoolbox.envs.utils.wrappers import (
    add_RewardUncertaintyEnvClassWrapper,
)
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import (
    exploration,
    log,
    postprocessing,
    miscellaneous,
    plot,
    callbacks,
    cross_play,
    config_helper,
    exp_analysis,
)

logger = logging.getLogger(__name__)


def main(
    debug,
    train_n_replicates=None,
    filter_utilitarian=None,
    env=None,
    use_r2d2=False,
    use_policy_gratient=False,
    hyperparameter_search=False,
):
    hparams = get_hyperparameters(
        debug,
        train_n_replicates,
        filter_utilitarian,
        env,
        use_r2d2=use_r2d2,
        use_policy_gratient=use_policy_gratient,
        hyperparameter_search=hyperparameter_search,
    )

    if hparams["load_plot_data"] is None:
        ray.init(
            num_gpus=0,
            num_cpus=os.cpu_count(),
            local_mode=hparams["debug"],
        )

        # Train
        if hparams["load_policy_data"] is None:
            experiment_analysis_per_welfare = train_for_each_welfare_function(
                hparams
            )
        else:
            experiment_analysis_per_welfare = load_experiment_analysis(
                hparams["load_policy_data"]
            )
        # Eval & Plot
        analysis_metrics_per_mode = config_and_evaluate_cross_play(
            experiment_analysis_per_welfare, hparams
        )

        ray.shutdown()
    else:
        experiment_analysis_per_welfare = None
        # Plot
        analysis_metrics_per_mode = config_and_evaluate_cross_play(
            experiment_analysis_per_welfare, hparams
        )

    return experiment_analysis_per_welfare, analysis_metrics_per_mode


def get_hyperparameters(
    debug,
    train_n_replicates=None,
    filter_utilitarian=None,
    env=None,
    reward_uncertainty=0.0,
    use_r2d2=False,
    use_policy_gratient=False,
    hyperparameter_search=False,
):
    if not debug:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--env",
            type=str,
            choices=[
                "IteratedPrisonersDilemma",
                "IteratedAsymBoS",
                "IteratedAsymBoSandPD",
                "CoinGame",
                "ABCoinGame",
            ],
            help="Env to use.",
        )
        parser.add_argument("--train_n_replicates", type=int)
        args = parser.parse_args()
        args = args.__dict__
        if "env" in args.keys():
            env = args["env"]
        if "train_n_replicates" in args.keys():
            train_n_replicates = args["train_n_replicates"]
    print("env", env)
    if hyperparameter_search:
        if train_n_replicates is None:
            train_n_replicates = 1
        n_times_more_utilitarians_seeds = 1
    elif debug:
        train_n_replicates = 2
        n_times_more_utilitarians_seeds = 1
    else:
        if train_n_replicates is None:
            train_n_replicates = 40
            # train_n_replicates = 2
        n_times_more_utilitarians_seeds = 4

    n_seeds_to_prepare = train_n_replicates * (
        1 + n_times_more_utilitarians_seeds
    )
    pool_of_seeds = miscellaneous.get_random_seeds(n_seeds_to_prepare)
    exp_name, _ = log.log_in_current_day_dir("amTFT")
    hparams = {
        "debug": debug,
        "use_r2d2": use_r2d2,
        "filter_utilitarian": filter_utilitarian
        if filter_utilitarian is not None
        else not debug,
        "seeds": pool_of_seeds,
        "train_n_replicates": train_n_replicates,
        "n_times_more_utilitarians_seeds": n_times_more_utilitarians_seeds,
        "exp_name": exp_name,
        "log_n_points": 250,
        "num_envs_per_worker": 16,
        "load_plot_data": None,
        "load_policy_data": None,
        "amTFTPolicy": amTFT.AmTFTRolloutsTorchPolicy,
        "welfare_functions": [
            (postprocessing.WELFARE_INEQUITY_AVERSION, "inequity_aversion"),
            (postprocessing.WELFARE_UTILITARIAN, "utilitarian"),
        ],
        # "amTFT_punish_instead_of_selfish": False,
        # "use_short_debit_rollout": False,
        # "punishment_helped": False,
        "amTFT_punish_instead_of_selfish": True,
        "use_short_debit_rollout": True,
        "punishment_helped": True,
        "jitter": 0.05,
        "hiddens": [64],
        "gamma": 0.96,
        # If not in self play then amTFT
        # will be evaluated against a naive selfish policy or an exploiter
        "self_play": True,
        # "env_name": "IteratedPrisonersDilemma" if env is None else env,
        # "env_name": "IteratedBoS" if env is None else env,
        # "env_name": "IteratedAsymBoS" if env is None else env,
        # "env_name": "IteratedAsymBoSandPD" if env is None else env,
        "env_name": "CoinGame" if env is None else env,
        # "env_name": "ABCoinGame" if env is None else env,
        "overwrite_reward": True,
        "explore_during_evaluation": True,
        "reward_uncertainty": reward_uncertainty,
        "use_other_play": False,
        # "use_other_play": True,
        "use_policy_gratient": use_policy_gratient,
        "use_MSE_in_r2d2": True,
        "hyperparameter_search": hyperparameter_search,
        "using_wandb": use_policy_gratient,
    }

    if hparams["load_policy_data"] is not None:
        hparams["train_n_replicates"] = len(
            hparams["load_policy_data"]["Util"]
        )

    hparams = modify_hyperparams_for_the_selected_env(hparams)

    return hparams


def load_experiment_analysis(grouped_checkpoints_paths: dict):
    experiment_analysis = {}
    msg = "start load_experiment_analysis"
    print(msg)
    logger.info(msg)
    for group_name, checkpoints_paths in grouped_checkpoints_paths.items():
        one_experiment_analysis = (
            exp_analysis.load_experiment_analysis_wt_ckpt_only(
                checkpoints_paths, n_dir_level_between_ckpt_and_exp_state=3
            )
        )
        experiment_analysis[group_name] = one_experiment_analysis
    msg = "end load_experiment_analysis"
    print(msg)
    logger.info(msg)
    return experiment_analysis


def modify_hyperparams_for_the_selected_env(hp):
    hp["plot_keys"] = (
        amTFT.PLOT_KEYS + aggregate_and_plot_tensorboard_data.PLOT_KEYS
    )
    hp["plot_assemblage_tags"] = (
        amTFT.PLOT_ASSEMBLAGE_TAGS
        + aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
    )

    hp["buf_frac"] = 0.125
    hp["training_intensity"] = 1 if hp["debug"] else 40
    hp["rollout_length"] = 20
    hp["n_rollout_replicas"] = 10
    hp["beta_steps_config"] = [
        (0, 0.125),
        (1.0, 0.25),
    ]

    if "CoinGame" in hp["env_name"]:

        hp["n_steps_per_epi"] = 20 if hp["debug"] else 100
        hp["n_epi"] = 10 if hp["debug"] else 4000
        hp["eval_over_n_epi"] = 1
        # hp["base_lr"] = 0.1

        hp["bs_epi_mul"] = 4
        hp["both_players_can_pick_the_same_coin"] = False
        hp["sgd_momentum"] = 0.9

        hp["lambda"] = 0.96
        hp["alpha"] = 0.0
        hp["beta"] = config_helper.configurable_linear_scheduler(
            "beta_steps_config"
        )

        hp["debit_threshold"] = 3.0
        hp["punishment_multiplier"] = 6.0
        hp["jitter"] = 0.0
        hp["filter_utilitarian"] = False

        # hp["buf_frac"] = 0.5
        hp["target_network_update_freq"] = 30 * hp["n_steps_per_epi"]
        hp["last_exploration_temp_value"] = 0.003 / 4

        # hp["lr_steps_config"] = [
        #     (0, 1.0),
        #     (0.25, 0.5),
        #     (1.0, 1e-9),
        # ]
        # hp["temperature_steps_config"] = [
        #     (0, 0.75),
        #     (0.2, 0.45),
        #     (0.9, hp["last_exploration_temp_value"]),
        # ]

        hp["lr_steps_config"] = [
            (0, 1e-9),
            (0.2, 1.0),
            # (0.5, 1.0/3),
            # (0.5, 1.0/10),
            # (0.75, 1.0/10/3),
            (1.0, 1e-9),
            # (1.0, 1.0),
            # (1.0, 0.33),
        ]

        hp["buf_frac"] = 0.1
        # hp["last_exploration_temp_value"] = 0.1
        hp["temperature_steps_config"] = [
            (0, 0.75),
            # (0, 0.25),
            (0.2, 0.25),
            # (0.5, 0.1),
            # (0.7, 0.1),
            (0.9, hp["last_exploration_temp_value"]),
        ]

        # hp["buf_frac"] = 0.25
        hp["base_lr"] = 0.1 / 4

        if "ABCoinGame" in hp["env_name"]:
            raise NotImplementedError()
        else:
            hp["plot_keys"] += coin_game.PLOT_KEYS
            hp["plot_assemblage_tags"] += coin_game.PLOT_ASSEMBLAGE_TAGS
            hp["x_limits"] = (-0.1, 0.6)
            hp["y_limits"] = (-0.1, 0.6)
            hp["env_class"] = coin_game.CoinGame
    else:

        hp["plot_keys"] += matrix_sequential_social_dilemma.PLOT_KEYS
        hp[
            "plot_assemblage_tags"
        ] += matrix_sequential_social_dilemma.PLOT_ASSEMBLAGE_TAGS

        hp["base_lr"] = 0.03
        hp["bs_epi_mul"] = 4
        hp["n_steps_per_epi"] = 10 if hp["debug"] else 20
        hp["n_epi"] = 5 if hp["debug"] else 800
        hp["eval_over_n_epi"] = 5
        hp["lambda"] = 0.96
        hp["alpha"] = 0.0
        hp["beta"] = 1.0
        hp["sgd_momentum"] = 0.0

        hp["debit_threshold"] = 10.0
        hp["punishment_multiplier"] = 3.0

        hp["target_network_update_freq"] = 30 * hp["n_steps_per_epi"]
        hp["last_exploration_temp_value"] = 0.1

        hp["temperature_steps_config"] = [
            (0, 2.0),
            (0.33, 0.5),
            (0.66, hp["last_exploration_temp_value"]),
        ]
        hp["lr_steps_config"] = [
            (0, 0.0),
            (0.05, 1.0),
            (1.0, 1e-9),
        ]

        if "IteratedPrisonersDilemma" == hp["env_name"]:
            hp["filter_utilitarian"] = False
            hp["x_limits"] = (-3.5, 0.5)
            hp["y_limits"] = (-3.5, 0.5)
            hp["utilitarian_filtering_threshold"] = -2.5
            hp[
                "env_class"
            ] = matrix_sequential_social_dilemma.IteratedPrisonersDilemma
        elif "IteratedAsymBoS" == hp["env_name"]:
            hp["x_limits"] = (-0.1, 4.1)
            hp["y_limits"] = (-0.1, 4.1)
            hp["utilitarian_filtering_threshold"] = 3.2
            hp["env_class"] = matrix_sequential_social_dilemma.IteratedAsymBoS
        elif "IteratedAsymBoSandPD" == hp["env_name"]:
            # hp["x_limits"] = (-3.1, 4.1)
            # hp["y_limits"] = (-3.1, 4.1)
            hp["x_limits"] = (-6.1, 5.1)
            hp["y_limits"] = (-6.1, 5.1)
            hp["utilitarian_filtering_threshold"] = 3.2
            hp[
                "env_class"
            ] = matrix_sequential_social_dilemma.IteratedAsymBoSandPD
        elif "IteratedBoS" == hp["env_name"]:
            hp["x_limits"] = (-0.1, 3.1)
            hp["y_limits"] = (-0.1, 3.1)
            hp["utilitarian_filtering_threshold"] = 2.6
            hp["env_class"] = matrix_sequential_social_dilemma.IteratedBoS
        else:
            raise NotImplementedError(f'hp["env_name"]: {hp["env_name"]}')

    hp["plot_axis_scale_multipliers"] = (
        (1 / hp["n_steps_per_epi"]),  # for x axis
        (1 / hp["n_steps_per_epi"]),
    )  # for y axis

    if "reward_uncertainty" in hp.keys() and hp["reward_uncertainty"] != 0.0:
        hp["env_class"] = add_RewardUncertaintyEnvClassWrapper(
            env_class=hp["env_class"],
            reward_uncertainty_std=hp["reward_uncertainty"],
        )

    hp["temperature_schedule"] = config_helper.get_temp_scheduler()
    hp["lr_schedule"] = config_helper.get_lr_scheduler()

    return hp


def train_for_each_welfare_function(hp):
    experiment_analysis_per_welfare = {}

    if hp["use_r2d2"]:
        trainer = dqn.r2d2.R2D2Trainer
    elif hp["use_policy_gratient"]:
        trainer = PPOTrainer
    else:
        trainer = dqn.DQNTrainer

    for welfare_fn, welfare_group_name in hp["welfare_functions"]:
        print("==============================================")
        print(
            "Going to start two_steps_training with welfare function",
            welfare_fn,
        )
        if welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            hp = preprocess_utilitarian_config(hp)
        stop, env_config, rllib_config = get_rllib_config(hp, welfare_fn)

        rllib_config_copy = copy.deepcopy(rllib_config)
        if hp["using_wandb"]:
            rllib_config_copy["logger_config"]["wandb"][
                "group"
            ] += f"_{welfare_group_name}"

        exp_name = os.path.join(hp["exp_name"], welfare_fn)
        results = amTFT.train_amtft(
            stop_config=stop,
            rllib_config=rllib_config_copy,
            name=exp_name,
            TrainerClass=trainer,
            plot_keys=hp["plot_keys"],
            plot_assemblage_tags=hp["plot_assemblage_tags"],
            debug=hp["debug"],
            log_to_file=not hp["debug"],
            punish_instead_of_selfish=hp["amTFT_punish_instead_of_selfish"],
            loggers=DEFAULT_LOGGERS + (WandbLogger,)
            if hp["using_wandb"]
            else DEFAULT_LOGGERS,
        )
        if welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            results, hp = postprocess_utilitarian_results(
                results, env_config, hp
            )
        experiment_analysis_per_welfare[welfare_group_name] = results
    return experiment_analysis_per_welfare


def preprocess_utilitarian_config(hp):
    hp_copy = copy.deepcopy(hp)
    if hp_copy["filter_utilitarian"]:
        hp_copy["train_n_replicates"] = (
            hp_copy["train_n_replicates"]
            * hp_copy["n_times_more_utilitarians_seeds"]
        )
    return hp_copy


def get_rllib_config(hp, welfare_fn, eval=False):
    stop_config = {
        "episodes_total": hp["n_epi"],
    }

    env_config = get_env_config(hp)

    policies = get_policies(hp, env_config, welfare_fn, eval)

    selected_seeds = hp["seeds"][: hp["train_n_replicates"]]
    hp["seeds"] = hp["seeds"][hp["train_n_replicates"] :]

    rllib_config = {
        "env": hp["env_class"],
        "env_config": env_config,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
            "observation_fn": amTFT.observation_fn,
        },
        "gamma": hp["gamma"],
        "seed": tune.grid_search(selected_seeds),
        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": hp["base_lr"],
        # Learning rate schedule
        "lr_schedule": hp["lr_schedule"],
        # If not None, clip gradients during optimization at this value
        "grad_clip": 1,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": hp["n_steps_per_epi"],
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": tune.sample_from(
            lambda spec: int(
                spec.config["env_config"]["max_steps"]
                * spec.config["env_config"]["bs_epi_mul"]
            )
        ),
        "training_intensity": tune.sample_from(
            lambda spec: spec.config["num_envs_per_worker"]
            * max(1, spec.config["num_workers"])
            * hp["training_intensity"]
        ),
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": hp["n_steps_per_epi"]
        if hp["debug"]
        else int(hp["n_steps_per_epi"] * hp["n_epi"] / hp["log_n_points"]),
        "min_iter_time_s": 0.0,
        # General config
        "framework": "torch",
        "num_workers": 0,
        # LE supports only 1 env per worker only otherwise
        # several episodes would be played at the same time
        "num_envs_per_worker": hp["num_envs_per_worker"],
        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and
        # `examples/custom_metrics_and_callbacks.py` for more usage
        # information.
        "callbacks": callbacks.merge_callbacks(
            amTFT.AmTFTCallbacks,
            log.get_logging_callbacks_class(
                log_full_epi=hp["num_envs_per_worker"] == 1,
                log_model_sumamry=True,
            ),
        ),
        # === DQN Models ===
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": hp["target_network_update_freq"],
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": tune.sample_from(
            lambda spec: max(
                int(
                    spec.config["env_config"]["max_steps"]
                    * spec.config["env_config"]["buf_frac"]
                    * spec.stop["episodes_total"]
                ),
                5,
            )
        ),
        # Whether to use dueling dqn
        "dueling": True,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": hp["hiddens"],
        # Whether to use double dqn
        "double_q": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        "model": {
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": [64],
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "relu",
        },
        # How many steps of the model to sample before learning starts.
        "learning_starts": tune.sample_from(
            lambda spec: int(
                spec.config["env_config"]["max_steps"]
                * spec.config["env_config"]["bs_epi_mul"]
            )
        ),
        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case,
            # this is the name (str) of any class present in the
            # `rllib.utils.exploration` package.
            # You can also provide the python class directly or
            # the full location of your class (e.g.
            # "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": exploration.SoftQSchedule,
            # Add constructor kwargs here (if any).
            "temperature_schedule": hp["temperature_schedule"],
        },
        "optimizer": {
            "sgd_momentum": hp["sgd_momentum"],
        },
        # "log_level": "DEBUG",
        "evaluation_interval": None,
        "evaluation_parallel_to_training": False,
    }

    rllib_config = _modify_config_for_coin_game(rllib_config, env_config, hp)
    rllib_config, stop_config = _modify_config_for_r2d2(
        rllib_config, hp, stop_config, eval
    )
    rllib_config, stop_config = _modify_config_for_policy_gradient(
        rllib_config, hp, stop_config, eval
    )
    return stop_config, env_config, rllib_config


def get_env_config(hp):
    if "CoinGame" in hp["env_name"]:
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hp["n_steps_per_epi"],
            "grid_size": 3,
            "both_players_can_pick_the_same_coin": hp[
                "both_players_can_pick_the_same_coin"
            ],
            "punishment_helped": hp["punishment_helped"],
        }
    else:
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hp["n_steps_per_epi"],
        }
    env_config["bs_epi_mul"] = hp.get("bs_epi_mul", None)
    env_config["buf_frac"] = hp.get("buf_frac", None)
    env_config["temperature_steps_config"] = hp.get(
        "temperature_steps_config", None
    )
    env_config["lr_steps_config"] = hp.get("lr_steps_config", None)
    env_config["beta_steps_config"] = hp.get("beta_steps_config", None)
    env_config["use_other_play"] = hp["use_other_play"]
    return env_config


def get_policies(hp, env_config, welfare_fn, eval=False):
    policy_class = hp["amTFTPolicy"]
    (
        nested_policy_class,
        coop_nested_policy_class,
        nested_selfish_policy_class,
    ) = get_nested_policy_class(hp)

    if eval:
        nested_policy_class = coop_nested_policy_class

    amTFT_config_update = merge_dicts(
        amTFT.DEFAULT_CONFIG,
        {
            # Set to True to train the nested policies and to False to use them
            "working_state": "train_coop",
            "welfare_key": welfare_fn,
            "verbose": 1 if hp["debug"] else 0,
            # "verbose": 2,
            "punishment_multiplier": hp["punishment_multiplier"],
            "debit_threshold": hp["debit_threshold"],
            "rollout_length": 2
            if hp["debug"]
            else min(hp["n_steps_per_epi"], hp["rollout_length"]),
            "n_rollout_replicas": 2
            if hp["debug"]
            else hp["n_rollout_replicas"],
            "punish_instead_of_selfish": hp["amTFT_punish_instead_of_selfish"],
            "use_short_debit_rollout": hp["use_short_debit_rollout"],
            "optimizer": {
                "sgd_momentum": hp["sgd_momentum"],
            },
            "nested_policies": [
                {
                    "Policy_class": coop_nested_policy_class,
                    "config_update": {
                        postprocessing.ADD_INEQUITY_AVERSION_WELFARE: [
                            welfare_fn
                            == postprocessing.WELFARE_INEQUITY_AVERSION,
                            hp["alpha"],
                            hp["beta"],
                            hp["gamma"],
                            hp["lambda"],
                        ],
                        postprocessing.ADD_UTILITARIAN_WELFARE: (
                            welfare_fn == postprocessing.WELFARE_UTILITARIAN
                        ),
                    },
                },
                {"Policy_class": nested_policy_class, "config_update": {}},
                {
                    "Policy_class": coop_nested_policy_class,
                    "config_update": {},
                },
                {"Policy_class": nested_policy_class, "config_update": {}},
            ],
        },
    )

    if hp["amTFT_punish_instead_of_selfish"]:
        amTFT_config_update["nested_policies"].append(
            {"Policy_class": nested_selfish_policy_class, "config_update": {}},
        )

    policy_1_config = copy.deepcopy(amTFT_config_update)
    policy_1_config["own_policy_id"] = env_config["players_ids"][0]
    policy_1_config["opp_policy_id"] = env_config["players_ids"][1]

    policy_2_config = copy.deepcopy(amTFT_config_update)
    policy_2_config["own_policy_id"] = env_config["players_ids"][1]
    policy_2_config["opp_policy_id"] = env_config["players_ids"][0]

    policies = {
        env_config["players_ids"][0]: [
            # The default policy is DQN defined in DQNTrainer but
            # we overwrite it to use the LE policy
            policy_class,
            hp["env_class"](env_config).OBSERVATION_SPACE,
            hp["env_class"].ACTION_SPACE,
            policy_1_config,
        ],
        env_config["players_ids"][1]: [
            policy_class,
            hp["env_class"](env_config).OBSERVATION_SPACE,
            hp["env_class"].ACTION_SPACE,
            policy_2_config,
        ],
    }

    return policies


def get_nested_policy_class(hp):
    nested_selfish_policy_class = _select_base_policy(hp)

    if hp["use_policy_gratient"]:
        original_postprocess_fn = compute_gae_for_sample_batch
    else:
        original_postprocess_fn = postprocess_nstep_and_prio

    coop_nested_policy_class = nested_selfish_policy_class.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.welfares_postprocessing_fn(),
            original_postprocess_fn,
        )
    )

    if hp["amTFT_punish_instead_of_selfish"]:
        nested_policy_class = nested_selfish_policy_class.with_updates(
            # TODO problem: this prevent to use HP searches on gamma etc.
            postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
                postprocessing.welfares_postprocessing_fn(
                    add_opponent_neg_reward=True,
                ),
                original_postprocess_fn,
            )
        )
    else:
        nested_policy_class = nested_selfish_policy_class
    return (
        nested_policy_class,
        coop_nested_policy_class,
        nested_selfish_policy_class,
    )


def _select_base_policy(hp):
    if hp["use_policy_gratient"]:
        print("using PPOTorchPolicy")
        assert not hp["use_r2d2"]
        # nested_policy_class = PPOTorchPolicy
        nested_policy_class = MyPPOTorchPolicy
    elif hp["use_r2d2"]:
        print("using augmented_r2d2.MyR2D2TorchPolicy")
        if hp["use_MSE_in_r2d2"]:
            nested_policy_class = augmented_r2d2.MyR2D2TorchPolicyWtMSELoss
        else:
            nested_policy_class = augmented_r2d2.MyR2D2TorchPolicy

    else:
        nested_policy_class = amTFT.DEFAULT_NESTED_POLICY_SELFISH

    # if hp["use_other_play"]:
    #     print("use_other_play with_updates")
    #
    #     if "CoinGame" in hp["env_name"]:
    #         symetries_available = coin_game.CoinGame.SYMMETRIES
    #     elif hp["env_name"] == "IteratedBoS":
    #         symetries_available = (
    #             matrix_sequential_social_dilemma.IteratedBoS.SYMMETRIES
    #         )
    #     else:
    #         raise NotImplementedError()
    #
    #     nested_policy_class = nested_policy_class.with_updates(
    #         _after_loss_init=partial(
    #             other_play.after_init_wrap_model_other_play,
    #             symetries_available=symetries_available,
    #         )
    #     )
    return nested_policy_class


def _modify_config_for_coin_game(rllib_config, env_config, hp):
    if "CoinGame" in hp["env_name"]:
        rllib_config["hiddens"] = [32]
        rllib_config["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1]],
            # [Channel, [Kernel, Kernel], Stride]]
            "fcnet_hiddens": [64, 64],
        }
    return rllib_config


def _modify_config_for_r2d2(rllib_config, hp, stop_config, eval=False):
    if hp["use_r2d2"]:
        rllib_config["model"]["use_lstm"] = True
        rllib_config["use_h_function"] = False
        rllib_config["burn_in"] = 0
        rllib_config["zero_init_states"] = False
        rllib_config["model"]["lstm_cell_size"] = 16
        if hp["debug"]:
            rllib_config["model"]["max_seq_len"] = 2
        else:
            rllib_config["model"]["max_seq_len"] = 20
            rllib_config["env_config"]["bs_epi_mul"] = 4
            if "CoinGame" in hp["env_name"]:
                rllib_config["training_intensity"] = tune.sample_from(
                    lambda spec: spec.config["num_envs_per_worker"]
                    * max(1, spec.config["num_workers"])
                    * hp["training_intensity"]
                )
                if not eval:
                    stop_config["episodes_total"] = 8000 * hp["n_epi"] / 4000

    return rllib_config, stop_config


def _modify_config_for_policy_gradient(
    rllib_config, hp, stop_config, eval=False
):
    if hp["use_policy_gratient"]:
        # rllib_config.pop("lr_schedule")
        rllib_config.pop("target_network_update_freq")
        rllib_config.pop("buffer_size")
        rllib_config.pop("dueling")
        rllib_config.pop("double_q")
        rllib_config.pop("prioritized_replay")
        rllib_config.pop("training_intensity")
        rllib_config.pop("hiddens")
        rllib_config.pop("learning_starts")

        if hp["debug"]:
            rllib_config["train_batch_size"] = int(128 * 2)
            rllib_config["num_sgd_iter"] = 2
        elif not hp["debug"] and not eval:
            # if hp["hyperparameter_search"]:
            #     rllib_config["train_batch_size"] = tune.grid_search(
            #         [1024, 4096]
            #     )
            # else:
            rllib_config["train_batch_size"] = 4096

            stop_config["episodes_total"] = 5000 * hp["n_epi"] / 4000

        rllib_config["exploration_config"] = {
            # The Exploration class to use. In the simplest case,
            # this is the name (str) of any class present in the
            # `rllib.utils.exploration` package.
            # You can also provide the python class directly or
            # the full location of your class (e.g.
            # "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": StochasticSampling,
            # Add constructor kwargs here (if any).
            # "temperature_schedule": hp["temperature_schedule"],
        }

        if hp["hyperparameter_search"]:
            # rllib_config["lr"] = 1e-4
            rllib_config["lr"] = tune.grid_search([1e-3, 3e-4])
            rllib_config["vf_loss_coeff"] = tune.grid_search([3.0, 1.0, 0.3])
            # rllib_config["model"]["vf_share_layers"] = tune.grid_search(
            #     [False, True]
            # )
            # rllib_config["use_gae"] = tune.grid_search([True, False])
            # rllib_config["batch_mode"] = tune.grid_search(
            #     ["truncate_episodes", "complete_episodes"]
            # )
            # rllib_config["batch_mode"] = "complete_episodes"

            rllib_config["env_config"]["beta_steps_config"] = [
                (0, 0.125 * 3),
                (1.0, 0.25 * 3),
            ]
            # rllib_config["env_config"]["beta_steps_config"] = [
            #     (0, 0.125*4),
            #     (1.0, 0.25*4),
            # ]

        else:
            rllib_config["lr"] = 0.1 / 300
        # hp["sgd_momentum"] = 0.5

        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        # "vf_loss_coeff": 1.0,
        # # Coefficient of the entropy regularizer.
        # "entropy_coeff": 0.0,
        # # Decay schedule for the entropy regularizer.
        # "entropy_coeff_schedule": None,
        # # PPO clip parameter.
        # "clip_param": 0.3,
        # # Clip param for the value function. Note that this is sensitive to the
        # # scale of the rewards. If your expected V is large, increase this.
        # "vf_clip_param": 10.0,
        # # If specified, clip the global norm of gradients by this amount.
        # "grad_clip": None,
        # # Target value for KL divergence.
        # "kl_target": 0.01,

        # if hp["hyperparameter_search"]:
        #     rllib_config["kl_target"] = tune.grid_search([0.003, 0.01, 0.03])
        # rllib_config["vf_clip_param"] = 1.0
        # rllib_config["vf_loss_coeff"] = 0.1

        rllib_config["logger_config"] = {
            "wandb": {
                "project": "amTFT",
                "group": hp["exp_name"],
                "api_key_file": os.path.join(
                    os.path.dirname(__file__), "../../api_key_wandb"
                ),
                "log_config": True,
            },
        }

    return rllib_config, stop_config


def postprocess_utilitarian_results(results, env_config, hp):
    """Reverse the changes made by preprocess_utilitarian_results"""

    hp_cp = copy.deepcopy(hp)

    if hp["filter_utilitarian"]:
        hp_cp["train_n_replicates"] = (
            hp_cp["train_n_replicates"]
            // hp_cp["n_times_more_utilitarians_seeds"]
        )
        results = exp_analysis.filter_trials(
            results,
            metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
            metric_threshold=hp_cp["utilitarian_filtering_threshold"]
            * hp_cp["n_steps_per_epi"],
            metric_mode="last-5-avg",
            threshold_mode="above",
        )
        if len(results.trials) > hp_cp["train_n_replicates"]:
            results.trials = results.trials[: hp_cp["train_n_replicates"]]
        elif len(results.trials) < hp_cp["train_n_replicates"]:
            print("WARNING: not enough Utilitarian trials above threshold!!!")
    return results, hp_cp


def config_and_evaluate_cross_play(experiment_analysis_per_welfare, hp):
    config_eval, env_config, stop, hp_eval = _generate_eval_config(hp)
    return evaluate_self_play_cross_play(
        experiment_analysis_per_welfare, config_eval, env_config, stop, hp_eval
    )


def evaluate_self_play_cross_play(
    experiment_analysis_per_welfare, config_eval, env_config, stop, hp_eval
):
    if hp_eval["use_r2d2"]:
        trainer = dqn.r2d2.R2D2Trainer
    elif hp_eval["use_policy_gratient"]:
        trainer = PPOTrainer
    else:
        trainer = dqn.DQNTrainer

    exp_name = os.path.join(hp_eval["exp_name"], "eval")
    evaluator = cross_play.evaluator.SelfAndCrossPlayEvaluator(
        exp_name=exp_name,
        local_mode=hp_eval["debug"],
    )
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=config_eval,
        stop_config=stop,
        policies_to_load_from_checkpoint=copy.deepcopy(
            env_config["players_ids"]
        ),
        experiment_analysis_per_welfare=experiment_analysis_per_welfare,
        rllib_trainer_class=trainer,
        n_self_play_per_checkpoint=hp_eval["n_self_play_per_checkpoint"],
        n_cross_play_per_checkpoint=hp_eval["n_cross_play_per_checkpoint"],
        to_load_path=hp_eval["load_plot_data"],
    )

    return plot_evaluation(
        hp_eval, evaluator, analysis_metrics_per_mode, env_config
    )


def plot_evaluation(hp_eval, evaluator, analysis_metrics_per_mode, env_config):
    if "CoinGame" in hp_eval["env_name"]:
        background_area_coord = None
    else:
        background_area_coord = hp_eval["env_class"].PAYOFF_MATRIX
    plot_config = plot.PlotConfig(
        xlim=hp_eval["x_limits"],
        ylim=hp_eval["y_limits"],
        markersize=5,
        alpha=1.0,
        jitter=hp_eval["jitter"],
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        plot_max_n_points=hp_eval["train_n_replicates"],
        x_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][0],
        y_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][1],
        background_area_coord=background_area_coord,
    )
    evaluator.plot_results(
        analysis_metrics_per_mode,
        plot_config=plot_config,
        x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}",
    )

    # print_inequity_aversion_welfare(env_config, analysis_metrics_per_mode)

    return analysis_metrics_per_mode


def _generate_eval_config(hp):
    hp_eval = modify_hp_for_evaluation(hp, hp["eval_over_n_epi"])
    fake_welfare_function = postprocessing.WELFARE_INEQUITY_AVERSION
    stop_config, env_config, rllib_config = get_rllib_config(
        hp_eval, fake_welfare_function, eval=True
    )
    config_eval, stop_config = modify_config_for_evaluation(
        rllib_config, hp_eval, env_config, stop_config
    )
    return config_eval, env_config, stop_config, hp_eval


def modify_hp_for_evaluation(hp: dict, eval_over_n_epi: int = 1):
    hp_eval = copy.deepcopy(hp)
    # TODO is the overwrite_reward hp useless?
    hp_eval["overwrite_reward"] = False
    hp_eval["num_envs_per_worker"] = 1
    hp_eval["n_epi"] = eval_over_n_epi
    if hp_eval["debug"]:
        hp_eval["n_epi"] = 1
        hp_eval["n_steps_per_epi"] = 5
    hp_eval["bs_epi_mul"] = 1
    hp_eval["plot_axis_scale_multipliers"] = (
        # for x axis
        (1 / hp_eval["n_steps_per_epi"]),
        # for y axis
        (1 / hp_eval["n_steps_per_epi"]),
    )
    hp_eval["n_self_play_per_checkpoint"] = 1
    hp_eval["n_cross_play_per_checkpoint"] = min(
        5,
        (
            (hp_eval["train_n_replicates"] * len(hp_eval["welfare_functions"]))
            - 1
        ),
    )
    return hp_eval


def modify_config_for_evaluation(config_eval, hp, env_config, stop_config):
    config_eval["explore"] = False
    config_eval.pop("seed")
    # = None
    config_eval["num_workers"] = 0
    assert (
        config_eval["num_envs_per_worker"] == 1
    ), f'num_envs_per_worker {config_eval["num_envs_per_worker"]}'
    assert (
        stop_config["episodes_total"] <= 20
    ), f'episodes_total {stop_config["episodes_total"]}'
    policies = config_eval["multiagent"]["policies"]
    for policy_id in policies.keys():
        policy_config = policies[policy_id][3]
        policy_config["working_state"] = "eval_amtft"
    if not hp["self_play"]:
        naive_player_id = env_config["players_ids"][-1]
        naive_player_policy_config = policies[naive_player_id][3]
        naive_player_policy_config["working_state"] = "eval_naive_selfish"

    if hp["explore_during_evaluation"] and not hp["use_policy_gratient"]:
        tmp_mul = 1.0
        config_eval["explore"] = (miscellaneous.OVERWRITE_KEY, True)
        config_eval["exploration_config"] = {
            "type": config_eval["exploration_config"]["type"],
            "temperature_schedule": PiecewiseSchedule(
                endpoints=[
                    (0, tmp_mul * hp["last_exploration_temp_value"]),
                    (0, tmp_mul * hp["last_exploration_temp_value"]),
                ],
                outside_value=tmp_mul * hp["last_exploration_temp_value"],
                framework="torch",
            ),
        }

    if hp["debug"] and hp.get("debit_threshold_debug_override", True):
        for policy_id in policies.keys():
            policies[policy_id][3]["debit_threshold"] = 0.5
            policies[policy_id][3]["last_k"] = hp["n_steps_per_epi"] - 1

    return config_eval, stop_config


def print_inequity_aversion_welfare(env_config, analysis_metrics_per_mode):
    plotter = cross_play.evaluator.SelfAndCrossPlayPlotter()
    plotter._reset(
        x_axis_metric=f"nested_policy/{env_config['players_ids'][0]}/worker_0/"
        f"policy_0/sum_over_epi_inequity_aversion_welfare",
        y_axis_metric=f"nested_policy/{env_config['players_ids'][1]}/worker_0/"
        f"policy_0/sum_over_epi_inequity_aversion_welfare",
        metric_mode="avg",
    )
    for mode_metric in analysis_metrics_per_mode:
        print("mode_metric", mode_metric[0], mode_metric[3])
        x, y = plotter._extract_x_y_points(mode_metric[1])
        print("x", x)
        print("y", y)


if __name__ == "__main__":
    use_r2d2 = True
    use_policy_gratient = False
    #
    # use_r2d2 = False
    # use_policy_gratient = True
    #
    debug_mode = False
    # debug_mode = True

    # hyperparameter_search = True
    hyperparameter_search = False

    main(
        debug_mode,
        use_r2d2=use_r2d2,
        use_policy_gratient=use_policy_gratient,
        hyperparameter_search=hyperparameter_search,
    )
