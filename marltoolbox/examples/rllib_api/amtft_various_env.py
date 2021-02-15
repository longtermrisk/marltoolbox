import copy
import os

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_stats, postprocess_nstep_and_prio
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.schedules import PiecewiseSchedule

import torch

from marltoolbox.envs import matrix_sequential_social_dilemma, coin_game
from marltoolbox.algos import amTFT
from marltoolbox.utils import same_and_cross_perf, exploration, log, \
    postprocessing, miscellaneous
from marltoolbox.utils.plot import PlotConfig


def modify_hp_for_selected_env(hp):
    if hp["env"] == matrix_sequential_social_dilemma.IteratedPrisonersDilemma:
        hp["n_epi"] = 10 if hp["debug"] else 400
        hp["base_lr"] = 0.01
        hp["x_limits"] = (-3.5, 0.5)
        hp["y_limits"] = (-3.5, 0.5)
    elif hp["env"] == matrix_sequential_social_dilemma.IteratedAsymChicken:
        hp["n_epi"] = 10 if hp["debug"] else 400
        hp["debit_threshold"] = 2.0
        hp["x_limits"] = (-11.0, 4.5)
        hp["y_limits"] = (-11.0, 4.5)
        hp["use_adam"] = True
        if hp["use_adam"]:
            hp["base_lr"] = 0.04
        else:
            hp["base_lr"] = 0.01 / 5
    elif hp["env"] in (matrix_sequential_social_dilemma.IteratedBoS, matrix_sequential_social_dilemma.IteratedAsymBoS):
        hp["n_epi"] = 10 if hp["debug"] else 800
        hp["base_lr"] = 0.01
        hp["x_limits"] = (-1.0, 5.0)
        hp["y_limits"] = (-1.0, 5.0)
    elif hp["env"] in [coin_game.CoinGame, coin_game.AsymCoinGame]:
        hp["n_epi"] = 10 if hp["debug"] else 4000
        hp["base_lr"] = 0.1
        hp["x_limits"] = (-1.0, 3.0)
        hp["y_limits"] = (-1.0, 1.0)
        hp["gamma"] = 0.9
        hp["lambda"] = 0.9
        hp["alpha"] = 0.0
        hp["beta"] = 0.5
        hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.50), 0.1)],
            outside_value=0.1,
            framework="torch")
        hp["debit_threshold"] = 2.0
        hp["jitter"] = 0.02
    else:
        raise NotImplementedError(f'hp["env"]: {hp["env"]}')

    hp["scale_multipliers"] = ((1 / hp["n_steps_per_epi"]),  # for x axis
                               (1 / hp["n_steps_per_epi"]))  # for y axis

    return hp


def train(hp):
    tune_analysis_per_welfare = {}
    for welfare_fn, welfare_group_name in hp['welfare_functions']:
        print("==============================================")
        print("Going to start two_steps_training with welfare function", welfare_fn)
        if hp["filter_utilitarian"] and welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            hp = preprocess_utilitarian_config(hp)
        stop, env_config, trainer_config_update = get_rllib_config(hp, welfare_fn)
        print("trainer_config_update", trainer_config_update)
        results = amTFT.two_steps_training(stop=stop,
                                           config=trainer_config_update,
                                           name=hp["exp_name"],
                                           TrainerClass=dqn.DQNTrainer)
        if hp["filter_utilitarian"] and welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            results = postprocess_utilitarian_results(results, env_config, hp)
        tune_analysis_per_welfare[welfare_group_name] = results
    return tune_analysis_per_welfare


def preprocess_utilitarian_config(hp):
    hp_copy = copy.deepcopy(hp)
    hp_copy['train_n_replicates'] = hp_copy['train_n_replicates'] * hp_copy["n_times_more_utilitarians_seeds"]
    return hp_copy


def get_rllib_config(hp, welfare_fn):
    stop = {
        "episodes_total": hp["n_epi"],
    }

    env_config = get_env_config(hp)
    policies = get_policies(hp, env_config, welfare_fn)

    selected_seeds = hp["seeds"][:hp["train_n_replicates"]]
    hp["seeds"] = hp["seeds"][hp["train_n_replicates"]:]

    trainer_config_update = {
        "env": hp["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        "gamma": hp["gamma"],
        "min_iter_time_s": hp["min_iter_time_s"],
        "seed": tune.grid_search(selected_seeds),

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": hp["base_lr"],
        # Learning rate schedule
        "lr_schedule": [(0, hp["base_lr"]),
                        (int(hp["n_steps_per_epi"] * hp["n_epi"]), hp["base_lr"] / 1e9)],
        # Adam epsilon hyper parameter
        # "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": 1,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": hp["n_steps_per_epi"],
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": int(hp["n_steps_per_epi"] * hp["bs_epi_mul"]),

        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": hp["n_steps_per_epi"],

        # General config
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # LE supports only 1 worker only otherwise it would be mixing several opponents trajectories
        "num_workers": 0,
        # LE supports only 1 env per worker only otherwise several episodes would be played at the same time
        "num_envs_per_worker": 1,

        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
        # for more usage information.
        "callbacks": amTFT.get_amTFTCallBacks(
            additionnal_callbacks=[log.get_logging_callbacks_class(),
                                   # This only overwrite the reward that is used for training not the one in the metrics
                                   postprocessing.OverwriteRewardWtWelfareCallback]),
        # "log_level": "INFO",

    }

    trainer_config_update.update({
        # === DQN Models ===
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": hp["n_steps_per_epi"],
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(hp["n_steps_per_epi"] * hp["n_epi"]) // 4,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": hp["hiddens"],
        # Whether to use double dqn
        "double_q": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        "model": {
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": hp["hiddens"],
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "relu",
        },

        # How many steps of the model to sample before learning starts.
        "learning_starts": int(hp["n_steps_per_epi"] * hp["bs_epi_mul"]),

        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": exploration.SoftQSchedule,
            # Add constructor kwargs here (if any).
            "temperature_schedule": hp["temperature_schedule"] or PiecewiseSchedule(
                endpoints=[
                    (0, 10.0),
                    (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.33), 1.0),
                    (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.66), 0.1)],
                outside_value=0.1,
                framework="torch"),
        },
    })

    if hp["env"] in [coin_game.CoinGame, coin_game.AsymCoinGame]:
        trainer_config_update["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],  # [Channel, [Kernel, Kernel], Stride]]
        }

    return stop, env_config, trainer_config_update


def get_env_config(hp):
    if hp["env"] in (
    matrix_sequential_social_dilemma.IteratedPrisonersDilemma, matrix_sequential_social_dilemma.IteratedBoSAndPD,
    matrix_sequential_social_dilemma.IteratedAsymChicken, matrix_sequential_social_dilemma.IteratedBoS,
    matrix_sequential_social_dilemma.IteratedAsymBoS):
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hp["n_steps_per_epi"],
            "get_additional_info": True,
        }
    elif hp["env"] in [coin_game.CoinGame, coin_game.AsymCoinGame]:
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hp["n_steps_per_epi"],
            "grid_size": 3,
            "get_additional_info": True,
        }
    else:
        raise NotImplementedError()
    return env_config


def get_nested_policy_class(hp, welfare_fn):
    NestedPolicyClass = dqn.DQNTorchPolicy

    get_vars = lambda policy: policy.q_func_vars

    if not hp["use_adam"]:
        def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
            return torch.optim.SGD(get_vars(policy), lr=policy.cur_lr, momentum=config["sgd_momentum"])

        NestedPolicyClass = NestedPolicyClass.with_updates(optimizer_fn=sgd_optimizer_dqn)

    if hp["debug"]:
        NestedPolicyClass = NestedPolicyClass.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats))

    CoopNestedPolicyClass = NestedPolicyClass.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.get_postprocessing_welfare_function(
                add_utilitarian_welfare=welfare_fn == postprocessing.WELFARE_UTILITARIAN,
                add_inequity_aversion_welfare=welfare_fn == postprocessing.WELFARE_INEQUITY_AVERSION,
                inequity_aversion_alpha=hp["alpha"], inequity_aversion_beta=hp["beta"],
                inequity_aversion_gamma=hp["gamma"], inequity_aversion_lambda=hp["lambda"],
            ),
            postprocess_nstep_and_prio
        )
    )
    return NestedPolicyClass, CoopNestedPolicyClass


def get_policies(hp, env_config, welfare_fn):
    PolicyClass = amTFT.amTFTTorchPolicy
    NestedPolicyClass, CoopNestedPolicyClass = get_nested_policy_class(hp, welfare_fn)

    amTFT_config_update = merge_dicts(
        amTFT.DEFAULT_CONFIG,
        {
            # Set to True to train the nested policies and to False to use them
            "working_state": "train_coop",
            "welfare": welfare_fn,
            "verbose": 1 if hp["debug"] else 0,

            "sgd_momentum": 0.9,
            'nested_policies': [
                {"Policy_class": CoopNestedPolicyClass, "config_update": {}},
                {"Policy_class": NestedPolicyClass, "config_update": {}},
                {"Policy_class": CoopNestedPolicyClass, "config_update": {}},
                {"Policy_class": NestedPolicyClass, "config_update": {}},
            ]
        }
    )

    policy_1_config = copy.deepcopy(amTFT_config_update)
    policy_1_config["own_policy_id"] = env_config["players_ids"][0]
    policy_1_config["opp_policy_id"] = env_config["players_ids"][1]
    policy_1_config["debit_threshold"] = hp["debit_threshold"]

    policy_2_config = copy.deepcopy(amTFT_config_update)
    policy_2_config["own_policy_id"] = env_config["players_ids"][1]
    policy_2_config["opp_policy_id"] = env_config["players_ids"][0]
    policy_2_config["debit_threshold"] = hp["debit_threshold"]

    policies = {
        env_config["players_ids"][0]: (
            # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
            PolicyClass,
            hp["env"](env_config).OBSERVATION_SPACE,
            hp["env"].ACTION_SPACE,
            policy_1_config
        ),
        env_config["players_ids"][1]: (
            PolicyClass,
            hp["env"](env_config).OBSERVATION_SPACE,
            hp["env"].ACTION_SPACE,
            policy_2_config
        ),
    }

    return policies


def postprocess_utilitarian_results(results, env_config, hp):
    results = miscellaneous.filter_tune_results(
        results,
        metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        metric_threshold=hp["utilitarian_filtering_threshold"] * hp["n_steps_per_epi"],
        metric_mode="last-5-avg", threshold_mode="above")
    if len(results.trials) > hp['train_n_replicates']:
        results.trials = results.trials[:hp['train_n_replicates']]
    elif len(results.trials) < hp['train_n_replicates']:
        print("WARNING: not enough Utilitarian trials above threshold!!!")
    return results


def evaluate_same_and_cross_perf(tune_analysis_per_welfare, hp):
    config_eval, env_config, stop, hp_eval = generate_eval_config(hp)

    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluator(exp_name=hp_eval["exp_name"])
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=config_eval, stop_config=stop,
        policies_to_load_from_checkpoint=copy.deepcopy(env_config["players_ids"]),
        tune_analysis_per_exp=tune_analysis_per_welfare,
        TrainerClass=dqn.DQNTrainer,
        n_cross_play_per_checkpoint=min(5, (hp_eval["train_n_replicates"] * len(hp_eval["welfare_functions"])) - 1),
        to_load_path=hp_eval["load_plot_data"])

    plot_config = PlotConfig(xlim=hp_eval["x_limits"], ylim=hp_eval["y_limits"],
                             markersize=5, alpha=1.0, jitter=hp_eval["jitter"],
                             xlabel="player 1 payoffs", ylabel="player 2 payoffs",
                             plot_max_n_points=hp_eval["train_n_replicates"],
                             title="cross and same-play performances: " + hp_eval['env'].NAME,
                             x_scale_multiplier=hp_eval["scale_multipliers"][0],
                             y_scale_multiplier=hp_eval["scale_multipliers"][1])
    evaluator.plot_results(analysis_metrics_per_mode, plot_config=plot_config,
                           x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
                           y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}")
    return analysis_metrics_per_mode


def generate_eval_config(hp):
    hp_eval = modify_hp_for_evaluation(hp)
    fake_welfare_function = hp_eval["welfare_functions"][0][0]
    stop, env_config, trainer_config_update = get_rllib_config(hp_eval, fake_welfare_function)
    config_eval = modify_config_for_evaluation(trainer_config_update, hp_eval, env_config)
    return config_eval, env_config, stop, hp_eval


def modify_hp_for_evaluation(hp):
    hp_eval = copy.deepcopy(hp)
    hp_eval["overwrite_reward"] = False
    hp_eval["n_epi"] = 1
    hp_eval["n_steps_per_epi"] = 5 if hp_eval["debug"] else 100
    hp_eval["bs_epi_mul"] = 1
    return hp_eval


def modify_config_for_evaluation(config_eval, hp, env_config):
    config_eval["explore"] = False
    config_eval["seed"] = None
    for policy_id in config_eval["multiagent"]["policies"].keys():
        config_eval["multiagent"]["policies"][policy_id][3]["working_state"] = "eval_amtft"
    if not hp["self_play"]:
        naive_player_id = env_config["players_ids"][-1]
        config_eval["multiagent"]["policies"][naive_player_id][3]["working_state"] = "eval_naive_selfish"
    return config_eval


def main(debug, train_n_replicates=None, filter_utilitarian=None):
    train_n_replicates = 1 if debug else train_n_replicates
    train_n_replicates = 40 if train_n_replicates is None else train_n_replicates
    n_times_more_utilitarians_seeds = 4
    pool_of_seeds = miscellaneous.get_random_seeds(train_n_replicates * (1 + n_times_more_utilitarians_seeds))
    exp_name, _ = log.log_in_current_day_dir("amTFT")
    hparams = {
        "debug": debug,
        "filter_utilitarian": filter_utilitarian if filter_utilitarian is not None else not debug,

        "train_n_replicates": train_n_replicates,
        "n_times_more_utilitarians_seeds": n_times_more_utilitarians_seeds,

        "load_plot_data": None,
        # Example: "load_plot_data": ".../SameAndCrossPlay_save.p",

        "exp_name": exp_name,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "welfare_functions": [(postprocessing.WELFARE_INEQUITY_AVERSION, "inequity_aversion"),
                              (postprocessing.WELFARE_UTILITARIAN, "utilitarian")],
        "seeds": pool_of_seeds,

        "gamma": 0.5,
        "lambda": 0.9,
        "alpha": 0.0,
        "beta": 1.0,

        "temperature_schedule": False,
        "debit_threshold": 4.0,
        "jitter": 0.05,
        "hiddens": [64],

        # If not in self play then amTFT will be evaluated against a naive selfish policy
        "self_play": True,
        # "self_play": False, # Not tested

        "env": matrix_sequential_social_dilemma.IteratedPrisonersDilemma,
        "utilitarian_filtering_threshold": -2.5,
        # "env": matrix_sequential_social_dilemma.IteratedAsymBoS,
        # "utilitarian_filtering_threshold": 3.2,
        # "env": matrix_sequential_social_dilemma.IteratedAsymChicken,
        # "utilitarian_filtering_threshold": ...,
        # "env": coin_game.CoinGame
        # "env": coin_game.AsymCoinGame
        # "utilitarian_filtering_threshold": ...,

        # For training speed
        "min_iter_time_s": 0.0 if debug else 3.0,

        "overwrite_reward": True,
        "use_adam": False,
    }

    hparams = modify_hp_for_selected_env(hparams)

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=hparams["debug"])

        # Train
        tune_analysis_per_welfare = train(hparams)
        # Eval & Plot
        analysis_metrics_per_mode = evaluate_same_and_cross_perf(tune_analysis_per_welfare, hparams)

        ray.shutdown()
    else:
        tune_analysis_per_welfare = None
        # Plot
        analysis_metrics_per_mode = evaluate_same_and_cross_perf(tune_analysis_per_welfare, hparams)

    return tune_analysis_per_welfare, analysis_metrics_per_mode


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
