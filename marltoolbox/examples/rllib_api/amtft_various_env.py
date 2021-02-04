import copy

import os
import ray
from ray import tune
from ray.rllib.agents import dqn, a3c
from ray.rllib.agents.a3c import a3c_torch_policy
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_stats, postprocess_nstep_and_prio
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule

torch, nn = try_import_torch()

from marltoolbox.envs import matrix_sequential_social_dilemma, coin_game
from marltoolbox.algos import amTFT
from marltoolbox.utils import same_and_cross_perf, exploration, log, \
    postprocessing, policy, miscellaneous


def modify_hp_for_selected_env(hp):
    if hp["env"] == matrix_sequential_social_dilemma.IteratedPrisonersDilemma:
        hp["n_epi"] = 10 if hp["debug"] else 400
        hp["base_lr"] = 0.01
        hp["x_limits"] = ((-3.5, 0.5),)
        hp["y_limits"] = ((-3.5, 0.5),)
    elif hp["env"] == matrix_sequential_social_dilemma.IteratedAsymChicken:
        hp["n_epi"] = 10 if hp["debug"] else 400
        hp["debit_threshold"] = 2.0
        hp["x_limits"] = ((-11.0, 4.5),)
        hp["y_limits"] = ((-11.0, 4.5),)
        hp["use_adam"] = True
        if hp["use_adam"]:
            hp["base_lr"] = 0.04
        else:
            hp["base_lr"] = 0.01 / 5
    elif hp["env"] in (matrix_sequential_social_dilemma.IteratedBoS, matrix_sequential_social_dilemma.IteratedAsymBoS):
        hp["n_epi"] = 10 if hp["debug"] else 800
        hp["base_lr"] = 0.01
        hp["x_limits"] = ((-1.0, 5.0),)
        hp["y_limits"] = ((-1.0, 5.0),)
    elif hp["env"] in [coin_game.CoinGame, coin_game.AsymCoinGame]:
        hp["n_epi"] = 10 if hp["debug"] else 4000
        hp["base_lr"] = 0.1
        hp["x_limits"] = ((-1.0, 3.0),)
        hp["y_limits"] = ((-1.0, 1.0),)
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

    hp["scale_multipliers"] = (
        ((1 / hp["n_steps_per_epi"]),  # First metric, x scale multiplier
         (1 / hp["n_steps_per_epi"])),  # First metric, y scale multiplier
    )

    return hp


def get_env_config(hp):
    if hp["env"] in (matrix_sequential_social_dilemma.IteratedPrisonersDilemma, matrix_sequential_social_dilemma.IteratedBoSAndPD,
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
    NestedPolicyClass = hp["NestedPolicyClass"]

    if NestedPolicyClass == dqn.DQNTorchPolicy:
        additional_fn = postprocess_nstep_and_prio
        stats_fn = build_q_stats
        get_vars = lambda policy: policy.q_func_vars

    elif NestedPolicyClass == policy.A2CTorchPolicy:
        additional_fn = a3c.a3c_torch_policy.add_advantages
        stats_fn = a3c_torch_policy.loss_and_entropy_stats
        hp["base_lr"] = hp["base_lr"] * hp["a2c_lr_multiplier"]
        get_vars = lambda policy: policy.model.parameters()
    else:
        raise NotImplementedError()

    if not hp["use_adam"]:
        def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
            return torch.optim.SGD(get_vars(policy), lr=policy.cur_lr, momentum=config["sgd_momentum"])

        NestedPolicyClass = NestedPolicyClass.with_updates(optimizer_fn=sgd_optimizer_dqn)

    if hp["debug"]:
        NestedPolicyClass = NestedPolicyClass.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(stats_fn))

    CoopNestedPolicyClass = NestedPolicyClass.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.get_postprocessing_welfare_function(
                add_utilitarian_welfare=welfare_fn == postprocessing.WELFARE_UTILITARIAN,
                add_inequity_aversion_welfare=welfare_fn == postprocessing.WELFARE_INEQUITY_AVERSION,
                inequity_aversion_alpha=hp["alpha"], inequity_aversion_beta=hp["beta"],
                inequity_aversion_gamma=hp["gamma"], inequity_aversion_lambda=hp["lambda"],
            ),
            additional_fn
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
                                   postprocessing.OverwriteRewardWtWelfareCallback]),
        # "log_level": "INFO",

    }

    if hp["NestedPolicyClass"] == dqn.DQNTorchPolicy:
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
    # elif hp["NestedPolicyClass"] == a3c_torch_policy.A3CTorchPolicy:
    elif hp["NestedPolicyClass"] == policy.A2CTorchPolicy:

        trainer_config_update.update({
            # === A3C Settings ===
            # Should use a critic as a baseline (otherwise don't use value baseline;
            # required for using GAE).
            "use_critic": True,
            # If true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": False,
            # Size of rollout batch
            # "rollout_fragment_length": 10,
            # GAE(gamma) parameter
            "lambda": 1.0,
            # Max global norm for each gradient calculated by worker
            # "grad_clip": 40.0,
            # # Learning rate
            # "lr": 0.0001,
            # # Learning rate schedule
            # "lr_schedule": None,
            # Value Function Loss coefficient
            "vf_loss_coeff": 0.5,
            # Entropy coefficient
            "entropy_coeff": 1.0 * 3.0,
            # Min time per iteration
            # "min_iter_time_s": 5,
            # Workers sample async. Note that this increases the effective
            # rollout_fragment_length by up to 5x due to async buffering of batches.
            "sample_async": True,

            "lr": hp["base_lr"],
            # Learning rate schedule
            "lr_schedule": [(0, hp["base_lr"]),
                            (int(hp["n_steps_per_epi"] * hp["n_epi"]), hp["base_lr"] / 1e9)],

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
                "type": "StochasticSampling",
            },
        })
    else:
        raise NotImplementedError()

    if hp["env"] in [coin_game.CoinGame, coin_game.AsymCoinGame]:
        trainer_config_update["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],  # [Channel, [Kernel, Kernel], Stride]]
        }

    return stop, env_config, trainer_config_update


def preprocess_utilitarian_config(hp):
    hp_copy = copy.deepcopy(hp)
    hp_copy['train_n_replicates'] = hp_copy['train_n_replicates'] * hp_copy["n_times_more_utilitarians_seeds"]
    return hp_copy


def postprocess_utilitarian_results(results, env_config, hp):
    results = miscellaneous.filter_tune_results(
        results,
        metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        metric_threshold=3.2 * hp["n_steps_per_epi"],
        metric_mode="last-5-avg", threshold_mode="above")
    if len(results.trials) > hp['train_n_replicates']:
        results.trials = results.trials[:hp['train_n_replicates']]
    elif len(results.trials) < hp['train_n_replicates']:
        print("WARNING: not enough Utilitarian trials above threshold!!!")
    return results


def train(hp):
    results_list = []
    for welfare_fn in hp['welfare_functions']:
        print("==============================================")
        print("Going to start two_steps_training with welfare function", welfare_fn)
        if hp["filter_utilitarian"] and welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            hp = preprocess_utilitarian_config(hp)
        stop, env_config, trainer_config_update = get_rllib_config(hp, welfare_fn)
        print("trainer_config_update", trainer_config_update)
        results = amTFT.two_steps_training(stop=stop,
                                           config=trainer_config_update,
                                           name=hp["exp_name"],
                                           TrainerClass=hp["TrainerClass"])
        if hp["filter_utilitarian"] and welfare_fn == postprocessing.WELFARE_UTILITARIAN:
            results = postprocess_utilitarian_results(results, env_config, hp)
        results_list.append(results)
    return results_list


def evaluate_same_and_cross_perf(config_eval, results_list, hp, env_config, stop, train_n_replicates):
    # Evaluate
    config_eval["explore"] = False
    config_eval["seed"] = None
    for policy_id in config_eval["multiagent"]["policies"].keys():
        config_eval["multiagent"]["policies"][policy_id][3]["working_state"] = "eval_amtft"
    policies_to_load = copy.deepcopy(env_config["players_ids"])
    if not hp["self_play"]:
        naive_player_id = env_config["players_ids"][-1]
        config_eval["multiagent"]["policies"][naive_player_id][3]["working_state"] = "eval_naive_selfish"

    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluation(
        TrainerClass=hp["TrainerClass"],
        group_names=hp["group_names"],
        evaluation_config=config_eval,
        stop_config=stop,
        exp_name=hp["exp_name"],
        policies_to_train=["None"],
        policies_to_load_from_checkpoint=policies_to_load,
    )

    if hp["load_plot_data"] is None:
        analysis_metrics_per_mode = evaluator.perf_analysis(n_same_play_per_checkpoint=1,
                                                            n_cross_play_per_checkpoint=min(5, train_n_replicates - 1),
                                                            extract_checkpoints_from_results=results_list,
                                                            )
    else:
        analysis_metrics_per_mode = evaluator.load_results(to_load_path=hp["load_plot_data"])

    evaluator.plot_results(
        analysis_metrics_per_mode, title_sufix=": " + hp['env'].NAME,
        metrics=((f"policy_reward_mean/{env_config['players_ids'][0]}",
                  f"policy_reward_mean/{env_config['players_ids'][1]}"),),
        x_limits=hp["x_limits"], y_limits=hp["y_limits"],
        scale_multipliers=hp["scale_multipliers"], markersize=5, alpha=1.0, jitter=hp["jitter"],
        xlabel="player 1 payoffs", ylabel="player 2 payoffs", add_title=False, frameon=True,
        show_groups=True, plot_max_n_points=train_n_replicates
    )
    return analysis_metrics_per_mode


def main(debug, train_n_replicates=None, filter_utilitarian=True):
    train_n_replicates = 1 if debug else train_n_replicates
    train_n_replicates = 40 if train_n_replicates is None else train_n_replicates
    n_times_more_utilitarians_seeds = 4
    pool_of_seeds = miscellaneous.get_random_seeds(train_n_replicates*(1+n_times_more_utilitarians_seeds))
    exp_name, _ = log.log_in_current_day_dir("amTFT")
    hparams = {
        "debug": debug,
        "filter_utilitarian": filter_utilitarian,

        "train_n_replicates": train_n_replicates,
        "n_times_more_utilitarians_seeds": n_times_more_utilitarians_seeds,

        "NestedPolicyClass": dqn.DQNTorchPolicy,
        "TrainerClass": dqn.DQNTrainer,

        # The training hyperparameters with A2C are not tuned well, DQN instead is well tuned
        # "NestedPolicyClass": policy.A2CTorchPolicy,
        # "TrainerClass": a3c.A2CTrainer,
        # "a2c_lr_multiplier": 1.0/9.0,

        "load_plot_data": None,
        # Example: "load_plot_data": ".../SameAndCrossPlay_save.p",

        "exp_name": exp_name,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "welfare_functions": [postprocessing.WELFARE_INEQUITY_AVERSION, postprocessing.WELFARE_UTILITARIAN],
        "group_names": ["inequity_aversion", "utilitarian"],
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
        # "env": matrix_sequential_social_dilemma.IteratedAsymBoS,
        # "env": matrix_sequential_social_dilemma.IteratedAsymChicken,
        # "env": coin_game.CoinGame
        # "env": coin_game.AsymCoinGame

        # For training speed
        "min_iter_time_s": 0.0 if debug else 3.0,

        "overwrite_reward": True,
        "use_adam": False,
    }

    hparams = modify_hp_for_selected_env(hparams)

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0)

        # Train
        results_list = train(hparams)
        # Eval & Plot
        hparams["overwrite_reward"] = False
        hparams["n_epi"] = 1
        hparams["n_steps_per_epi"] = 5 if hparams["debug"] else 100
        hparams["bs_epi_mul"] = 1
        stop, env_config, trainer_config_update = get_rllib_config(hparams, hparams["welfare_functions"][0])
        analysis_metrics_per_mode = evaluate_same_and_cross_perf(trainer_config_update, results_list,
                                     hparams, env_config, stop, train_n_replicates)

        ray.shutdown()
    else:
        hparams["overwrite_reward"] = False
        hparams["n_epi"] = 1
        hparams["n_steps_per_epi"] = 5 if hparams["debug"] else 100
        hparams["bs_epi_mul"] = 1
        # Plot
        results_list = None
        stop, env_config, trainer_config_update = get_rllib_config(hparams, hparams["welfare_functions"][0])
        analysis_metrics_per_mode = evaluate_same_and_cross_perf(trainer_config_update, results_list,
                                     hparams, env_config, stop, train_n_replicates)

    return results_list, analysis_metrics_per_mode

if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
