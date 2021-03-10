import copy
import os

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn_torch_policy import \
    postprocess_nstep_and_prio
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.schedules import PiecewiseSchedule

from marltoolbox.algos import amTFT
from marltoolbox.envs import IteratedPrisonersDilemma, IteratedAsymBoS, \
    IteratedAsymChicken, CoinGame, AsymCoinGame
from marltoolbox.utils import exploration, log, \
    postprocessing, miscellaneous, plot, self_and_cross_perf


def main(debug, train_n_replicates=None, filter_utilitarian=None, env=None):
    hparams = get_hyperparameters(debug, train_n_replicates,
                                  filter_utilitarian, env)

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(),
                 num_gpus=0,
                 local_mode=hparams["debug"])

        # Train
        if hparams["load_policy_data"] is None:
            tune_analysis_per_welfare = \
                train_for_each_welfare_function(hparams)
        else:
            tune_analysis_per_welfare = \
                load_tune_analysis(hparams["load_policy_data"])
        # Eval & Plot
        analysis_metrics_per_mode = \
            evaluate_self_and_cross_perf(tune_analysis_per_welfare, hparams)

        ray.shutdown()
    else:
        tune_analysis_per_welfare = None
        # Plot
        analysis_metrics_per_mode = \
            evaluate_self_and_cross_perf(tune_analysis_per_welfare, hparams)

    return tune_analysis_per_welfare, analysis_metrics_per_mode


def get_hyperparameters(debug, train_n_replicates=None,
                        filter_utilitarian=None, env=None):
    if debug:
        train_n_replicates = 2
    elif train_n_replicates is None:
        train_n_replicates = 40

    n_times_more_utilitarians_seeds = 4
    n_seeds_to_prepare = \
        train_n_replicates * (1 + n_times_more_utilitarians_seeds)
    pool_of_seeds = miscellaneous.get_random_seeds(n_seeds_to_prepare)
    exp_name, _ = log.log_in_current_day_dir("amTFT")
    hparams = {
        "debug": debug,
        "filter_utilitarian":
            filter_utilitarian
            if filter_utilitarian is not None
            else not debug,
        "seeds": pool_of_seeds,
        "train_n_replicates": train_n_replicates,
        "n_times_more_utilitarians_seeds": n_times_more_utilitarians_seeds,
        "exp_name": exp_name,

        "load_plot_data": None,
        # Example: "load_plot_data": ".../SelfAndCrossPlay_save.p",

        "load_policy_data": None,
        # "load_policy_data": {
        #     "Util": [
        #         ".../IBP/amTFT/trials/"
        #         "DQN_AsymCoinGame_...",
        #         ".../IBP/amTFT/trials/"
        #         "DQN_AsymCoinGame_..."],
        #     'IA':[
        #         ".../temp/IBP/amTFT/trials/"
        #         "DQN_AsymCoinGame_...",
        #         ".../IBP/amTFT/trials/"
        #         "DQN_AsymCoinGame_..."],
        # },

        "amTFTPolicy": amTFT.amTFTRolloutsTorchPolicy,
        "welfare_functions": [
            (postprocessing.WELFARE_INEQUITY_AVERSION, "inequity_aversion"),
            (postprocessing.WELFARE_UTILITARIAN, "utilitarian")],
        "bs_epi_mul": 4,
        "temperature_schedule": False,
        "jitter": 0.05,
        "hiddens": [64],
        "base_lr": 0.01,

        # If not in self play then amTFT
        # will be evaluated against a naive selfish policy or an exploiter
        "self_play": True,
        # "self_play": False, # Not tested

        "env": "IteratedPrisonersDilemma",
        # "env": "IteratedAsymBoS",
        # "env": "IteratedAsymChicken",
        # "env": "CoinGame",
        # "env": "AsymCoinGame",

        "overwrite_reward": True,
        "explore_during_evaluation": True,

        # For training speed
        "min_iter_time_s": 0.0 if debug else 10.0,
    }

    if env is not None:
        hparams["env"] = env

    hparams = modify_hyperparams_for_the_selected_env(hparams)

    return hparams


def load_tune_analysis(grouped_checkpoints_paths: dict):
    tune_analysis = {}
    for group_name, checkpoints_paths in grouped_checkpoints_paths.items():
        one_tune_analysis = miscellaneous.load_one_tune_analysis(
            checkpoints_paths)
        tune_analysis[group_name] = one_tune_analysis
    return tune_analysis


def modify_hyperparams_for_the_selected_env(hp):
    # default values
    hp["last_exploration_temp_value"] = 0.1
    hp["gamma"] = 0.5
    hp["lambda"] = 0.9
    hp["alpha"] = 0.0
    hp["beta"] = 1.0
    hp["punishment_multiplier"] = 6.0
    hp["debit_threshold"] = 4.0
    hp["n_steps_per_epi"] = 20

    if "IteratedPrisonersDilemma" in hp["env"]:
        hp["n_epi"] = 10 if hp["debug"] else 400
        hp["x_limits"] = (-3.5, 0.5)
        hp["y_limits"] = (-3.5, 0.5)
        hp["utilitarian_filtering_threshold"] = -2.5
        hp["env"] = IteratedPrisonersDilemma
    elif "IteratedAsymChicken" in hp["env"]:
        hp["n_epi"] = 10 if hp["debug"] else 400
        hp["debit_threshold"] = 2.0
        hp["x_limits"] = (-11.0, 4.5)
        hp["y_limits"] = (-11.0, 4.5)
        hp["utilitarian_filtering_threshold"] = None
        hp["env"] = IteratedAsymChicken
        raise NotImplementedError(
            "utilitarian_filtering_threshold must have a value")
    elif "IteratedAsymBoS" in hp["env"]:
        hp["n_epi"] = 10 if hp["debug"] else 800
        hp["x_limits"] = (-0.1, 4.1)
        hp["y_limits"] = (-0.1, 4.1)
        hp["utilitarian_filtering_threshold"] = 3.2
        hp["env"] = IteratedAsymBoS
    elif "CoinGame" in hp["env"]:
        hp["n_epi"] = 10 if hp["debug"] else 4000
        hp["n_steps_per_epi"] = 20 if hp["debug"] else 100
        hp["base_lr"] *= 10
        if "AsymCoinGame" in hp["env"]:
            hp["x_limits"] = (-0.5, 3.0)
            hp["y_limits"] = (-1.1, 0.6)
            hp["env"] = AsymCoinGame
        else:
            hp["x_limits"] = (-0.5, 0.6)
            hp["y_limits"] = (-0.5, 0.6)
            hp["env"] = CoinGame
        hp["gamma"] = 0.9
        hp["lambda"] = 0.95
        hp["alpha"] = 0.0
        hp["beta"] = 0.5
        hp["last_exploration_temp_value"] = 0.1
        hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.50),
                 0.5),
                (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.75),
                 hp["last_exploration_temp_value"])],
            outside_value=hp["last_exploration_temp_value"],
            framework="torch")
        hp["debit_threshold"] = 4.0
        hp["jitter"] = 0.02
        hp["punishment_multiplier"] = 4.0
        hp["filter_utilitarian"] = False
        hp["both_players_can_pick_the_same_coin"] = False
    else:
        raise NotImplementedError(f'hp["env"]: {hp["env"]}')

    hp["plot_axis_scale_multipliers"] = (
        (1 / hp["n_steps_per_epi"]),  # for x axis
        (1 / hp["n_steps_per_epi"]))  # for y axis

    return hp


def train_for_each_welfare_function(hp):
    tune_analysis_per_welfare = {}
    for welfare_fn, welfare_group_name in hp['welfare_functions']:
        print("==============================================")
        print("Going to start two_steps_training with welfare function",
              welfare_fn)
        if welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            hp = preprocess_utilitarian_config(hp)
        stop, env_config, trainer_config_update = \
            get_rllib_config(hp, welfare_fn)
        print("trainer_config_update", trainer_config_update)
        exp_name = os.path.join(hp["exp_name"], welfare_fn)
        results = amTFT.train_amTFT(stop=stop,
                                    config=trainer_config_update,
                                    name=exp_name,
                                    TrainerClass=dqn.DQNTrainer)
        if welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            results, hp = postprocess_utilitarian_results(results, env_config,
                                                          hp)
        tune_analysis_per_welfare[welfare_group_name] = results
    return tune_analysis_per_welfare


def preprocess_utilitarian_config(hp):
    hp_copy = copy.deepcopy(hp)
    if hp_copy["filter_utilitarian"]:
        hp_copy['train_n_replicates'] = \
            hp_copy['train_n_replicates'] * \
            hp_copy["n_times_more_utilitarians_seeds"]
    if "CoinGame" in hp["env"].NAME:
        hp_copy['n_steps_per_epi'] = hp_copy['n_steps_per_epi'] // 5
        hp_copy["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (int(hp_copy["n_steps_per_epi"] * hp_copy["n_epi"] * 0.50),
                 0.1),
                (int(hp_copy["n_steps_per_epi"] * hp_copy["n_epi"] * 0.75),
                 hp_copy["last_exploration_temp_value"])],
            outside_value=hp_copy["last_exploration_temp_value"],
            framework="torch")
        hp_copy["plot_axis_scale_multipliers"] = (
            (1 / hp_copy["n_steps_per_epi"]),  # for x axis
            (1 / hp_copy["n_steps_per_epi"]))  # for y axis
    return hp_copy


def get_rllib_config(hp, welfare_fn, eval=False):
    stop = {
        "episodes_total": hp["n_epi"],
    }

    env_config = get_env_config(hp)
    policies = get_policies(hp, env_config, welfare_fn, eval)

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
        "lr_schedule": [(0,
                         hp["base_lr"]),
                        (int(hp["n_steps_per_epi"] * hp["n_epi"]),
                         hp["base_lr"] / 1e9)],
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
        # LE supports only 1 worker only otherwise
        # it would be mixing several opponents trajectories
        "num_workers": 0,
        # LE supports only 1 env per worker only otherwise
        # several episodes would be played at the same time
        "num_envs_per_worker": 1,

        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and
        # `examples/custom_metrics_and_callbacks.py` for more usage
        # information.
        "callbacks": amTFT.get_amTFTCallBacks(
            additionnal_callbacks=[
                log.get_logging_callbacks_class(),
                # This only overwrite the reward that is used for training
                # not the one in the metrics
                postprocessing.OverwriteRewardWtWelfareCallback]),
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
            # The Exploration class to use. In the simplest case,
            # this is the name (str) of any class present in the
            # `rllib.utils.exploration` package.
            # You can also provide the python class directly or
            # the full location of your class (e.g.
            # "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": exploration.SoftQSchedule,
            # Add constructor kwargs here (if any).
            "temperature_schedule":
                hp["temperature_schedule"] or
                PiecewiseSchedule(
                    endpoints=[
                        (0, 10.0),
                        (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.33),
                         1.0),
                        (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.66),
                         hp["last_exploration_temp_value"])],
                    outside_value=hp["last_exploration_temp_value"],
                    framework="torch"
                ),
        },
    })

    if hp["env"] in [CoinGame, AsymCoinGame]:
        trainer_config_update["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
            # [Channel, [Kernel, Kernel], Stride]]
        }

    return stop, env_config, trainer_config_update


def get_env_config(hp):
    if hp["env"] in (IteratedPrisonersDilemma, IteratedAsymChicken,
                     IteratedAsymBoS):
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hp["n_steps_per_epi"],
        }
    elif hp["env"] in [CoinGame, AsymCoinGame]:
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hp["n_steps_per_epi"],
            "grid_size": 3,
            "both_players_can_pick_the_same_coin":
                hp["both_players_can_pick_the_same_coin"]
        }
    else:
        raise NotImplementedError()
    return env_config


def get_policies(hp, env_config, welfare_fn, eval=False):
    PolicyClass = hp["amTFTPolicy"]
    NestedPolicyClass, CoopNestedPolicyClass = \
        get_nested_policy_class(hp, welfare_fn)

    if eval:
        NestedPolicyClass = CoopNestedPolicyClass

    amTFT_config_update = merge_dicts(
        amTFT.DEFAULT_CONFIG,
        {
            # Set to True to train the nested policies and to False to use them
            "working_state": "train_coop",
            "welfare": welfare_fn,
            "verbose": 1 if hp["debug"] else 0,
            "punishment_multiplier": hp["punishment_multiplier"],
            "debit_threshold": hp["debit_threshold"],
            "rollout_length": min(hp["n_steps_per_epi"], 40),

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

    policy_2_config = copy.deepcopy(amTFT_config_update)
    policy_2_config["own_policy_id"] = env_config["players_ids"][1]
    policy_2_config["opp_policy_id"] = env_config["players_ids"][0]

    policies = {
        env_config["players_ids"][0]: (
            # The default policy is DQN defined in DQNTrainer but
            # we overwrite it to use the LE policy
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


def get_nested_policy_class(hp, welfare_fn):
    NestedPolicyClass = amTFT.DEFAULT_NESTED_POLICY_SELFISH
    CoopNestedPolicyClass = NestedPolicyClass.with_updates(
        # TODO problem: this prevent to use HP searches on gamma etc.
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.welfares_postprocessing_fn(
                add_utilitarian_welfare=
                (welfare_fn == postprocessing.WELFARE_UTILITARIAN),
                add_inequity_aversion_welfare=
                (welfare_fn == postprocessing.WELFARE_INEQUITY_AVERSION),
                inequity_aversion_alpha=hp["alpha"],
                inequity_aversion_beta=hp["beta"],
                inequity_aversion_gamma=hp["gamma"],
                inequity_aversion_lambda=hp["lambda"],
            ),
            postprocess_nstep_and_prio
        )
    )
    return NestedPolicyClass, CoopNestedPolicyClass


def postprocess_utilitarian_results(results, env_config, hp):
    hp_cp = copy.deepcopy(hp)

    if hp["filter_utilitarian"]:
        hp_cp['train_n_replicates'] = \
            hp_cp['train_n_replicates'] // \
            hp_cp["n_times_more_utilitarians_seeds"]
        results = miscellaneous.filter_tune_results(
            results,
            metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
            metric_threshold=hp_cp["utilitarian_filtering_threshold"] *
                             hp_cp["n_steps_per_epi"],
            metric_mode="last-5-avg", threshold_mode="above")
        if len(results.trials) > hp_cp['train_n_replicates']:
            results.trials = results.trials[:hp_cp['train_n_replicates']]
        elif len(results.trials) < hp_cp['train_n_replicates']:
            print("WARNING: not enough Utilitarian trials above threshold!!!")

    if "CoinGame" in hp["env"].NAME:
        hp_cp['n_steps_per_epi'] = hp_cp['n_steps_per_epi'] * 5
        hp_cp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (int(hp_cp["n_steps_per_epi"] * hp_cp["n_epi"] * 0.50), 0.1),
                (int(hp_cp["n_steps_per_epi"] * hp_cp["n_epi"] * 0.75),
                 hp_cp["last_exploration_temp_value"])],
            outside_value=hp_cp["last_exploration_temp_value"],
            framework="torch")
        hp_cp["plot_axis_scale_multipliers"] = ((1 / hp_cp["n_steps_per_epi"]),
                                                (1 / hp_cp["n_steps_per_epi"]))
    return results, hp_cp


def evaluate_self_and_cross_perf(tune_analysis_per_welfare, hp):
    config_eval, env_config, stop, hp_eval = generate_eval_config(hp)

    exp_name = os.path.join(hp_eval["exp_name"], "eval")
    evaluator = self_and_cross_perf.SelfAndCrossPlayEvaluator(
        exp_name=exp_name)
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=config_eval,
        stop_config=stop,
        policies_to_load_from_checkpoint=copy.deepcopy(
            env_config["players_ids"]),
        tune_analysis_per_exp=tune_analysis_per_welfare,
        TrainerClass=dqn.DQNTrainer,
        n_cross_play_per_checkpoint=
        min(5,
            (hp_eval["train_n_replicates"] *
             len(hp_eval["welfare_functions"])) - 1),
        to_load_path=hp_eval["load_plot_data"])

    if hp["env"] in [CoinGame, AsymCoinGame]:
        background_area_coord = None
    else:
        background_area_coord = hp['env'].PAYOUT_MATRIX
    plot_config = plot.PlotConfig(
        xlim=hp_eval["x_limits"],
        ylim=hp_eval["y_limits"],
        markersize=5,
        alpha=1.0,
        jitter=hp_eval["jitter"],
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        plot_max_n_points=hp_eval["train_n_replicates"],
        # title="cross and same-play performances: " + hp_eval['env'].NAME,
        x_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][0],
        y_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][1],
        background_area_coord=background_area_coord
    )
    evaluator.plot_results(
        analysis_metrics_per_mode,
        plot_config=plot_config,
        x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}")

    print_inequity_aversion_welfare(env_config, analysis_metrics_per_mode)

    return analysis_metrics_per_mode


def generate_eval_config(hp):
    hp_eval = modify_hp_for_evaluation(hp)
    fake_welfare_function = postprocessing.WELFARE_INEQUITY_AVERSION
    stop, env_config, trainer_config_update = get_rllib_config(
        hp_eval,
        fake_welfare_function,
        eval=True)
    config_eval = modify_config_for_evaluation(
        trainer_config_update, hp_eval, env_config)
    return config_eval, env_config, stop, hp_eval


def modify_hp_for_evaluation(hp):
    hp_eval = copy.deepcopy(hp)
    hp_eval["overwrite_reward"] = False
    hp_eval["n_epi"] = 1
    hp_eval["n_steps_per_epi"] = 5 if hp_eval["debug"] else 100
    hp_eval["bs_epi_mul"] = 1
    hp_eval["plot_axis_scale_multipliers"] = (
        # for x axis
        (1 / hp_eval["n_steps_per_epi"]),
        # for y axis
        (1 / hp_eval["n_steps_per_epi"])
    )
    return hp_eval


def modify_config_for_evaluation(config_eval, hp, env_config):
    config_eval["explore"] = False
    config_eval["seed"] = None
    policies = config_eval["multiagent"]["policies"]
    for policy_id in policies.keys():
        policy_config = policies[policy_id][3]
        policy_config["working_state"] = "eval_amtft"
    if not hp["self_play"]:
        naive_player_id = env_config["players_ids"][-1]
        naive_player_policy_config = policies[naive_player_id][3]
        naive_player_policy_config["working_state"] = "eval_naive_selfish"

    if hp["explore_during_evaluation"]:
        config_eval["explore"] = (miscellaneous.OVERWRITE_KEY, True)
        config_eval["exploration_config"] = {
            "type": "SoftQ",
            "temperature": hp["last_exploration_temp_value"],
        }

    if hp["debug"]:
        for policy_id in policies.keys():
            policies[policy_id][3]["debit_threshold"] = 0.5
            policies[policy_id][3]["last_k"] = hp["n_steps_per_epi"] - 1

    return config_eval


def print_inequity_aversion_welfare(env_config, analysis_metrics_per_mode):
    plotter = self_and_cross_perf.SelfAndCrossPlayPlotter()
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
    debug_mode = True
    main(debug_mode)
