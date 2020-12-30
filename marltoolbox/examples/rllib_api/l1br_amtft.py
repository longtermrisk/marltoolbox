import copy
from functools import partial
import os
import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, after_init
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule

torch, nn = try_import_torch()

from marltoolbox.envs import matrix_SSD, coin_game
from marltoolbox.algos import amTFT, population
from marltoolbox.utils import same_and_cross_perf, restore, exploration, log, \
    postprocessing, lvl1_best_response, miscellaneous
from marltoolbox.examples.rllib_api import amtft_various_env


def modify_hp_for_selected_env(hp):
    if hp["env"] == matrix_SSD.IteratedPrisonersDilemma:
        hp["n_epi"] = 10 if hp["debug"] else 400
        hp["base_lr"] = 0.01
        hp["x_limits"] = ((-3.5, 0.5),)
        hp["y_limits"] = ((-3.5, 0.5),)
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

def get_policies(hp, welfare_fn, env_config, amTFT_agents_idx):
    PolicyClass = amTFT.amTFTTorchPolicy
    NestedPolicyClass, CoopNestedPolicyClass = amtft_various_env.get_nested_policy_class(hp, welfare_fn)

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

    policy_2_config = copy.deepcopy(amTFT_config_update)
    policy_2_config["own_policy_id"] = env_config["players_ids"][1]
    policy_2_config["opp_policy_id"] = env_config["players_ids"][0]

    policies = {}
    for policy_idx, policy_id in enumerate(env_config["players_ids"]):
        if policy_idx in amTFT_agents_idx:
            policy_config = copy.deepcopy(amTFT_config_update)
            policy_config["own_policy_id"] = env_config["players_ids"][policy_idx]
            policy_config["opp_policy_id"] = env_config["players_ids"][(policy_idx + 1) % 2]
            policies[policy_id] = (
                PolicyClass,
                hp["env"](env_config).OBSERVATION_SPACE,
                hp["env"].ACTION_SPACE,
                policy_config
            )
        else:
            # after_init_fn = partial(
            #     miscellaneous.sequence_of_fn_wt_same_args(
            #             function_list=[after_init,
            #                            restore.after_init_load_policy_checkpoint]
            #         ))
            policies[policy_id] = (
                DQNTorchPolicy, #.with_updates(after_init=after_init_fn),
                hp["env"](env_config).OBSERVATION_SPACE,
                hp["env"].ACTION_SPACE,
                {}
            )
    return policies


def get_rllib_config(hp, welfare_fn, amTFT_agents_idx, lvl1=False):
    stop = {
        "episodes_total": 10 if hp['debug'] else hp["n_epi"],
    }

    env_config = amtft_various_env.get_env_config(hp)
    policies = get_policies(hp, welfare_fn, env_config, amTFT_agents_idx)

    trainer_config_update = {
        "env": hp["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        # === DQN Models ===
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": hp["n_steps_per_epi"],
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
        "hiddens": [64],
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

        "gamma": hp["gamma"],
        "min_iter_time_s": 0.0 if hp['debug'] else 3.0,
        # Can't restaure stuff with search
        "seed": tune.grid_search(list(range(hp["n_seeds_lvl1"] if lvl1 else hp["n_seeds_lvl0"]))),

        # "evaluation_num_episodes": 100,
        # "evaluation_interval": hparams["n_epi"],

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
        # How many steps of the model to sample before learning starts.
        "learning_starts": int(hp["n_steps_per_epi"] * hp["bs_epi_mul"]),
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": hp["n_steps_per_epi"],
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": int(hp["n_steps_per_epi"] * hp["bs_epi_mul"]),

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
        # "callbacks": DefaultCallbacks,
        "callbacks": amTFT.get_amTFTCallBacks(
            additionnal_callbacks=[log.get_logging_callbacks_class(),
                                   postprocessing.OverwriteRewardWtWelfareCallback,
                                   population.PopulationOfIdenticalAlgoCallBacks]),
        "log_level": "INFO",

    }

    if hp["env"] == coin_game.CoinGame:
        trainer_config_update["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],  # [Channel, [Kernel, Kernel], Stride]]
        }

    return stop, env_config, trainer_config_update


def train_lvl0_population(hp):
    stop, env_config, trainer_config_update = get_rllib_config(hp, hp['welfare_functions'][0], amTFT_agents_idx=[0, 1])
    results = amTFT.two_steps_training(stop=stop,
                                       config=trainer_config_update,
                                       name=hp["exp_name"],
                                       TrainerClass=hp["TrainerClass"])
    return results


def train_lvl1_agents(hp, results_list_lvl0):
    lvl0_policy_idx = 1
    lvl1_policy_idx = 0
    stop, env_config, rllib_config = get_rllib_config(hp, hp['welfare_functions'][0],
                                                               amTFT_agents_idx=[lvl0_policy_idx],
                                                               lvl1=True)
    lvl0_checkpoints = miscellaneous.extract_checkpoints(results_list_lvl0)
    lvl0_policy_id = env_config["players_ids"][lvl0_policy_idx]
    lvl1_policy_id = env_config["players_ids"][lvl1_policy_idx]

    lvl1_best_response.prepare_config_for_lvl1_training(
        config=rllib_config,
        lvl0_policy_id=lvl0_policy_id, lvl1_policy_id=lvl1_policy_id,
        select_n_lvl0_from_population=hp["n_seeds_lvl0"] // hp["n_seeds_lvl1"],
        n_lvl1_to_train=hp["n_seeds_lvl1"],
        overlapping_population=False, lvl0_checkpoints=lvl0_checkpoints)

    rllib_config["multiagent"]["policies"][lvl0_policy_id][3]["explore"] = False
    if hp["self_play"]:
        rllib_config["multiagent"]["policies"][lvl0_policy_id][3]["working_state"] = "eval_amtft"
    else:
        rllib_config["multiagent"]["policies"][lvl0_policy_id][3]["working_state"] = "eval_naive_selfish"
    results = ray.tune.run(hp['TrainerClass'], config=rllib_config,
                           stop=stop, name=hp["exp_name"],
                           checkpoint_at_end=True,
                           metric="episode_reward_mean", mode="max")

    return results


def evaluate_same_and_cross_perf(trainer_config_update, results_list, hp, env_config, stop):
    # Evaluate
    trainer_config_update["explore"] = False
    trainer_config_update["seed"] = None
    for policy_id in trainer_config_update["multiagent"]["policies"].keys():
        trainer_config_update["multiagent"]["policies"][policy_id][3]["working_state"] = "eval_amtft"
    policies_to_load = copy.deepcopy(env_config["players_ids"])
    if not hp["self_play"]:
        naive_player_id = env_config["players_ids"][-1]
        trainer_config_update["multiagent"]["policies"][naive_player_id][3]["working_state"] = "eval_naive_selfish"

    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluation(TrainerClass=hp['TrainerClass'],
                                                               group_names=hp["group_names"],
                                                               evaluation_config=trainer_config_update,
                                                               stop_config=stop,
                                                               exp_name=hp["exp_name"],
                                                               policies_to_train=["None"],
                                                               policies_to_load_from_checkpoint=policies_to_load,
                                                               )

    if hp["load_plot_data"] is None:
        analysis_metrics_per_mode = evaluator.perf_analysis(n_same_play_per_checkpoint=1,
                                                            n_cross_play_per_checkpoint=(hp["n_seeds_lvl1"] *
                                                                                         len(hp["group_names"])) - 1,
                                                            extract_checkpoints_from_results=results_list,
                                                            )
    else:
        analysis_metrics_per_mode = evaluator.load_results(to_load_path=hp["load_plot_data"])

    evaluator.plot_results(analysis_metrics_per_mode,
                           metrics=((f"policy_reward_mean/{env_config['players_ids'][0]}",
                                     f"policy_reward_mean/{env_config['players_ids'][1]}"),),
                           x_limits=hp["x_limits"], y_limits=hp["y_limits"],
                           scale_multipliers=hp["scale_multipliers"],
                           markersize=5,
                           alpha=1.0
                           )



def main(debug):
    exp_name, _ = log.log_in_current_day_dir("L1BR_amTFT")
    hparams = {
        "debug": debug,

        "load_plot_data": None,

        "exp_name": exp_name,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "welfare_functions": [postprocessing.WELFARE_UTILITARIAN],
        "group_names": ["utilitarian"],

        "n_seeds_lvl0": 2 if debug else 8,
        "n_seeds_lvl1": 1 if debug else 2,

        "gamma": 0.5,
        "lambda": 0.9,
        "alpha": 0.0,
        "beta": 1.0,
        "temperature_schedule": False,

        "self_play": True,
        # "self_play": False,

        "env": matrix_SSD.IteratedPrisonersDilemma,
        # "env": coin_game.CoinGame

        "NestedPolicyClass": dqn.DQNTorchPolicy,
        "TrainerClass": dqn.DQNTrainer,

        "use_adam": False,
    }

    hparams = modify_hp_for_selected_env(hparams)

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=4, num_gpus=1)

        # # Train
        results_list_lvl0 = train_lvl0_population(hp=hparams)
        results_list_lvl1 = train_lvl1_agents(hp=hparams, results_list_lvl0=results_list_lvl0, )
        results = results_list_lvl1

        # Eval
        results_list = [results]
        hparams["n_epi"] = 1
        hparams["n_steps_per_epi"] = 10 if hparams["debug"] else 100
        stop, env_config, trainer_config_update = get_rllib_config(hparams, hparams['welfare_functions'][0],
                                                                   amTFT_agents_idx=["player_col"])
        evaluate_same_and_cross_perf(trainer_config_update, results_list, hparams, env_config, stop)

        ray.shutdown()
    else:
        results_list = None
        hparams["n_epi"] = 1
        hparams["n_steps_per_epi"] = 10 if hparams["debug"] else 100
        stop, env_config, trainer_config_update = get_rllib_config(hparams, hparams['welfare_functions'][0],
                                                                   amTFT_agents_idx=["player_col"])
        evaluate_same_and_cross_perf(trainer_config_update, results_list, hparams, env_config, stop)


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
