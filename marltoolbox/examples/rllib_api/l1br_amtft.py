import copy

import os
import ray
from ray import tune
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_stats, DQNTorchPolicy
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule

torch, nn = try_import_torch()

from marltoolbox.envs import matrix_SSD, coin_game
from marltoolbox.algos import amTFT, population
from marltoolbox.utils import same_and_cross_perf, restore, exploration, log, \
    preprocessing, l1_best_response, miscellaneous


def get_config(hp, welfare_fn, amTFT_agents_idx, lvl1=False):
    stop = {
        "episodes_total": hp["n_epi"],
    }

    policyClass = amTFT.amTFTTorchPolicy

    hp["scale_multipliers"] = (
        ((1 / hp["n_steps_per_epi"]),  # First metric, x scale multiplier
         (1 / hp["n_steps_per_epi"])),  # First metric, y scale multiplier
    )

    if hp["env"] in (matrix_SSD.IteratedPrisonersDilemma, matrix_SSD.IteratedBoSAndPD, matrix_SSD.IteratedAsymChicken):
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hp["n_steps_per_epi"],
            "reward_randomness": 0.1,
            "get_additional_info": True,
        }

        amTFT_config_update = merge_dicts(
            amTFT.DEFAULT_CONFIG,
            {
                # Set to True to train the nested policies and to False to use them
                "working_state": "train_coop",
                "welfare": welfare_fn,
            }
        )
    elif hp["env"] == coin_game.CoinGame:
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hp["n_steps_per_epi"],
            "batch_size": 1,
            "grid_size": 3,
            "get_additional_info": True,
        }

        def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
            return torch.optim.SGD(policy.q_func_vars, lr=policy.cur_lr, momentum=config["sgd_momentum"])

        amTFT_config_update = merge_dicts(
            amTFT.DEFAULT_CONFIG,
            {
                # Set to True to train the nested policies and to False to use them
                "working_state": "train_coop",
                "welfare": welfare_fn,

                "sgd_momentum": 0.9,
                'nested_policies': [
                    {"Policy_class":
                         DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats),
                                                     optimizer_fn=sgd_optimizer_dqn),
                     "config_update": {}},
                    {"Policy_class":
                         DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats),
                                                     optimizer_fn=sgd_optimizer_dqn),
                     "config_update": {}},
                    {"Policy_class":
                         DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats),
                                                     optimizer_fn=sgd_optimizer_dqn),
                     "config_update": {}},
                    {"Policy_class":
                         DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats),
                                                     optimizer_fn=sgd_optimizer_dqn),
                     "config_update": {}},
                ]
            }
        )
    else:
        raise ValueError

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
            policy_config["opp_policy_id"] = env_config["players_ids"][(policy_idx+1)%2]
            policies[policy_id] = (
                policyClass,
                hp["env"](env_config).OBSERVATION_SPACE,
                hp["env"].ACTION_SPACE,
                policy_config
            )
        else:
            policies[policy_id] = (
                DQNTorchPolicy.with_updates(after_init=restore.after_init_load_policy_checkpoint),
                hp["env"](env_config).OBSERVATION_SPACE,
                hp["env"].ACTION_SPACE,
                {}
            )

    # TODO remove the useless hyper-parameters
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
        "buffer_size": int(hp["n_steps_per_epi"] * hp["n_epi"])//4,
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
        "min_iter_time_s": 0.33,
        # Can't restaure stuff with search
        # "seed": hp["seed"],
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
                    (int(hparams["n_steps_per_epi"] * hparams["n_epi"] * 0.33), 1.0),
                    (int(hparams["n_steps_per_epi"] * hparams["n_epi"] * 0.66), 0.1)],
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
        "callbacks": amTFT.get_amTFTCallBacks(add_utilitarian_welfare=True,
                                     add_inequity_aversion_welfare=True,
                                     inequity_aversion_alpha=hp["alpha"], inequity_aversion_beta=hp["beta"],
                                     inequity_aversion_gamma=hp["gamma"],
                                     inequity_aversion_lambda=hp["lambda"],
                                     additionnal_callbacks= [log.LoggingCallbacks, population.PopulationOfIdenticalAlgoCallBacks ]),
        # "callbacks": miscellaneous.merge_callbacks(
        #     amTFT.get_amTFTCallBacks(add_utilitarian_welfare=True,
        #                              add_inequity_aversion_welfare=True,
        #                              inequity_aversion_alpha=hp["alpha"], inequity_aversion_beta=hp["beta"],
        #                              inequity_aversion_gamma=hp["gamma"],
        #                              inequity_aversion_lambda=hp["lambda"]),
        #     log.LoggingCallbacks
        # ),

        "log_level": "INFO",

    }

    if hp["env"] == coin_game.CoinGame:
        trainer_config_update["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],  # [Channel, [Kernel, Kernel], Stride]]
        }

    return stop, env_config, trainer_config_update


def train_lvl0_population(hp):
    # results_list = []
    # for welfare_fn in hp['welfare_functions']:
    stop, env_config, trainer_config_update = get_config(hp, hp['welfare_functions'][0], amTFT_agents_idx=[0,1])
    results = amTFT.two_steps_training(stop=stop,
                                       config=trainer_config_update,
                                       name=hp["exp_name"])
    # results_list.append(results)
    # return results_list
    return results

def train_lvl1_agents(hp, results_list_lvl0):

    lvl0_policy_idx = 1
    lvl1_policy_idx = 0
    stop, env_config, trainer_config_update = get_config(hp, hp['welfare_functions'][0],
                                                         amTFT_agents_idx=[lvl0_policy_idx],
                                                         lvl1=True)
    lvl0_checkpoints = miscellaneous.extract_checkpoints(results_list_lvl0)
    lvl0_policy_id = env_config["players_ids"][lvl0_policy_idx]
    lvl1_policy_id = env_config["players_ids"][lvl1_policy_idx]

    l1_best_response.prepare_config_for_lvl1_training(
        config=trainer_config_update,
        lvl0_policy_id=lvl0_policy_id, lvl1_policy_id=lvl1_policy_id,
        select_n_lvl0_from_population=hp["n_seeds_lvl0"] // hp["n_seeds_lvl1"],
        n_lvl1_to_train=hp["n_seeds_lvl1"],
        overlapping_population=False, lvl0_checkpoints=lvl0_checkpoints)

    # Evaluate
    # trainer_config_update["explore"] = False
    # trainer_config_update["seed"] = None
    # for policy_id in trainer_config_update["multiagent"]["policies"].keys():
    trainer_config_update["multiagent"]["policies"][lvl0_policy_id][3]["working_state"] = "eval_amtft"
    trainer_config_update["multiagent"]["policies"][lvl0_policy_id][3]["explore"] = False
    if not hp["self_play"]:
        trainer_config_update["multiagent"]["policies"][lvl0_policy_id][3]["working_state"] = "eval_naive_selfish"

    results = ray.tune.run(amTFT.amTFTTrainer, config=trainer_config_update,
                           stop=stop, name=hp["exp_name"],
                           checkpoint_at_end=True,
                           metric="episode_reward_mean", mode="max")

    # results_list.append(results)
    # return results_list
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


    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluation(TrainerClass=amTFT.amTFTTrainer,
                                                                group_names=hp["group_names"],
                                                                evaluation_config=trainer_config_update,
                                                                stop_config=stop,
                                                                exp_name=hp["exp_name"],
                                                                policies_to_train=["None"],
                                                                policies_to_load_from_checkpoint=policies_to_load,
                                                                )

    if hparams["load_plot_data"] is None:
        analysis_metrics_per_mode = evaluator.perf_analysis(n_same_play_per_checkpoint=1,
                                                            n_cross_play_per_checkpoint=(hp["n_seeds_lvl1"]*
                                                                                         len(hp["group_names"]))-1,
                                                            extract_checkpoints_from_results=results_list,
                                                            )
    else:
        analysis_metrics_per_mode = evaluator.load_results(to_load_path=hparams["load_plot_data"])

    evaluator.plot_results(analysis_metrics_per_mode,
                           metrics=((f"policy_reward_mean/{env_config['players_ids'][0]}",
                                     f"policy_reward_mean/{env_config['players_ids'][1]}"),),
                           x_limits=hp["x_limits"], y_limits=hp["y_limits"],
                           scale_multipliers=hp["scale_multipliers"],
                           markersize=5,
                           alpha=1.0
    )

# TODO update the unique rollout worker after every episode
# TODO check than no bug arise from the fact that there is 2 policies
#  (one used to produce samples in the rolloutworker and one used to train the models)
if __name__ == "__main__":
    exp_name, _ = log.log_in_current_day_dir("amTFT")
    hparams = {
        "debug": True,

        "load_plot_data": None,

        "exp_name": exp_name,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "welfare_functions": [preprocessing.WELFARE_UTILITARIAN], # preprocessing.WELFARE_INEQUITY_AVERSION,
        "group_names": ["utilitarian"], # "inequity_aversion",

        "n_seeds_lvl0": 8,
        "n_seeds_lvl1": 2,

        "gamma": 0.5,
        "lambda": 0.9,
        "alpha": 0.0,
        "beta": 1.0,
        "temperature_schedule": False,

        "self_play": True,
        # "self_play": False,

        "env": matrix_SSD.IteratedPrisonersDilemma,
        # "env": matrix_SSD.BOTS_PD,
        # "env": matrix_SSD.IteratedAsymChicken,
        # "env": coin_game.CoinGame
    }

    if hparams["env"] == matrix_SSD.IteratedPrisonersDilemma:
        hparams["n_epi"] = 40 if hparams["debug"] else 400
        hparams["base_lr"] = 0.04
        hparams["x_limits"] = ((-3.5, 0.5),)
        hparams["y_limits"] = ((-3.5, 0.5),)
    elif hparams["env"] == matrix_SSD.IteratedBoSAndPD:
        hparams["n_epi"] = 40 if hparams["debug"] else 400
        hparams["base_lr"] = 0.04
        hparams["x_limits"] = ((-4.0, 4.0),)
        hparams["y_limits"] = ((-4.0, 4.0),)
    elif hparams["env"] == matrix_SSD.IteratedAsymChicken:
        hparams["n_epi"] = 40 if hparams["debug"] else 400
        hparams["base_lr"] = 0.04
        hparams["x_limits"] = ((-11.0, 4.0),)
        hparams["y_limits"] = ((-11.0, 4.0),)
    elif hparams["env"] == coin_game.CoinGame:
        hparams["n_epi"] = 40 if hparams["debug"] else 12000
        hparams["base_lr"] = 0.1
        hparams["x_limits"] = ((-2.5, 2.5),)
        hparams["y_limits"] = ((-2.5, 2.5),)
        hparams["gamma"] = 0.9
        hparams["lambda"] = 0.9
        hparams["alpha"] = 0.0
        hparams["beta"] = 0.5
        hparams["temperature_schedule"] = PiecewiseSchedule(
                endpoints=[
                    (0, 2.0),
                    (int(hparams["n_steps_per_epi"] * hparams["n_epi"] * 0.50), 0.1)],
                outside_value=0.1,
                framework="torch")

        # hparams["welfare_functions"].reverse()
        # hparams["group_names"].reverse()
    else:
        raise NotImplementedError(f'hparams["env"]: {hparams["env"]}')


    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=4, num_gpus=1)

        # # Train
        results_list_lvl0 = train_lvl0_population(hp=hparams)
        results_list_lvl1 = train_lvl1_agents(hp=hparams, results_list_lvl0=results_list_lvl0,)
        results = results_list_lvl1

        # Eval
        results_list = [results]
        hparams["n_epi"] = 1
        hparams["n_steps_per_epi"] = 10 if hparams["debug"] else 100
        # stop, env_config, trainer_config_update = get_config(hparams, hparams["welfare_functions"][0])
        stop, env_config, trainer_config_update = get_config(hparams, hparams['welfare_functions'][0],
                                                             amTFT_agents_idx=["player_col"])
        evaluate_same_and_cross_perf(trainer_config_update, results_list, hparams, env_config, stop)

        ray.shutdown()
    else:
        results_list = None
        hparams["n_epi"] = 1
        hparams["n_steps_per_epi"] = 10 if hparams["debug"] else 100
        stop, env_config, trainer_config_update = get_config(hparams, hparams['welfare_functions'][0],
                                                             amTFT_agents_idx=["player_col"])
        evaluate_same_and_cross_perf(trainer_config_update, results_list, hparams, env_config, stop)