##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the
# tune_class_api/lola_pg_official.py file
##########

import copy
import functools
import os
import time

import ray
import torch
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_torch_policy import (
    DQNTorchPolicy,
    build_q_stats,
    after_init,
)
from ray.rllib.utils.schedules import PiecewiseSchedule

from marltoolbox.algos.lola.train_cg_tune_class_API import LOLAPGCG
from marltoolbox.algos.lola.train_pg_tune_class_API import LOLAPGMatrice
from marltoolbox.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
    IteratedBoS,
    IteratedAsymChicken,
    IteratedAsymBoS,
)
from marltoolbox.envs.vectorized_coin_game import (
    VectorizedCoinGame,
    AsymVectorizedCoinGame,
)
from marltoolbox.utils import (
    policy,
    log,
    miscellaneous,
    lvl1_best_response,
    exploration,
    restore,
    callbacks,
)


# TODO make it work for all env (not only ACG and CG)? or only for them


def main(debug):
    n_in_lvl0_population = 2 if debug else 40
    n_lvl1 = 1 if debug else 1
    timestamp = int(time.time())
    lvl0_seeds = [
        seed + timestamp for seed in list(range(n_in_lvl0_population))
    ]
    lvl1_seeds = list(range(n_lvl1))

    exp_name, _ = log.log_in_current_day_dir("L1BR_LOLA_PG")

    tune_hparams = {
        "exp_name": exp_name,
        "load_data": None,
        # Example: "load_data": ".../lvl1_results.p",
        "load_population": None,
        # Example: "load_population":
        # [".../checkpoint.json", ".../checkpoint.json", ...]
        "num_episodes": 5 if debug else 2000,
        "trace_length": 5 if debug else 20,
        "lr": None,
        "gamma": 0.5,
        "batch_size": 5 if debug else 512,
        # "env_name": "IteratedPrisonersDilemma",
        # "env_name": "IteratedBoS",
        # "env_name": "IteratedAsymBoS",
        "env_name": "VectorizedCoinGame",
        # "env_name": "AsymVectorizedCoinGame",
        "pseudo": False,
        "grid_size": 3,
        "lola_update": True,
        "opp_model": False,
        "mem_efficient": True,
        "lr_correction": 1,
        "bs_mul": 1 / 10,
        "simple_net": True,
        "hidden": 32,
        "reg": 0,
        "set_zero": 0,
        "exact": False,
        "warmup": 1,
        "lvl0_seeds": lvl0_seeds,
        "lvl1_seeds": lvl1_seeds,
        "changed_config": False,
        "ac_lr": 1.0,
        "summary_len": 1,
        "use_MAE": False,
        "use_toolbox_env": True,
        "clip_loss_norm": False,
        "clip_lola_update_norm": False,
        "clip_lola_correction_norm": 3.0,
        "clip_lola_actor_norm": 10.0,
        "entropy_coeff": 0.001,
        "weigth_decay": 0.03,
        "lola_correction_multiplier": 1,
        "lr_decay": True,
        "correction_reward_baseline_per_step": False,
        "use_critic": False,
    }

    rllib_hparams = {
        "debug": debug,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "sgd_momentum": 0.9,
        "temperature_schedule": False,
    }

    if tune_hparams["load_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0)

        # # Train
        if tune_hparams["load_population"] is None:
            results_list_lvl0 = train_lvl0_population(tune_hp=tune_hparams)
            log.save_metrics(results_list_lvl0, exp_name, "lvl0_results.p")
        else:
            results_list_lvl0 = []

        results_list_lvl1 = train_lvl1_agents(
            tune_hp=tune_hparams,
            rllib_hp=rllib_hparams,
            results_list_lvl0=results_list_lvl0,
        )
        log.save_metrics(
            results_list_lvl1, exp_name, "lvl1_results.p", limit=True
        )

        ray.shutdown()
    else:
        # TODO print that every time, not only when loading
        log.pprint_saved_metrics(
            tune_hparams["load_data"],
            keywords_to_print=[
                "policy_reward_mean",
                "speed.*mean",
                "own.*mean",
                "analysis",
                "^avg$",
                "last-10-avg",
            ],
        )


def get_tune_config(hp: dict):
    tune_config = copy.deepcopy(hp)
    assert not tune_config["exact"]

    # Resolve default parameters
    if "CoinGame" in tune_config["env_name"]:
        if hp["env_name"] == "VectorizedCoinGame":
            tune_config["env_class"] = VectorizedCoinGame
        elif hp["env_name"] == "AsymVectorizedCoinGame":
            tune_config["env_class"] = AsymVectorizedCoinGame
        tune_config["num_episodes"] = (
            100000 if hp["num_episodes"] is None else hp["num_episodes"]
        )
        tune_config["trace_length"] = (
            150 if hp["trace_length"] is None else hp["trace_length"]
        )
        tune_config["batch_size"] = (
            4000 if hp["batch_size"] is None else hp["batch_size"]
        )
        tune_config["lr"] = 0.005 if hp["lr"] is None else hp["lr"]
        tune_config["gamma"] = 0.96 if hp["gamma"] is None else hp["gamma"]
        hp["x_limits"] = ((-1.0, 1.0),)
        hp["y_limits"] = ((-1.0, 1.0),)
        hp["jitter"] = 0.02
        hp["tune_class"] = LOLAPGCG
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "grid_size": tune_config["grid_size"],
            "get_additional_info": True,
            "both_players_can_pick_the_same_coin": False,
            "force_vectorize": False,
            "same_obs_for_each_player": True,
        }
        tune_config["metric"] = "player_blue_pick_speed"
    else:
        if hp["env_name"] == "IteratedPrisonersDilemma":
            tune_config["env_class"] = IteratedPrisonersDilemma
        elif hp["env_name"] == "IteratedAsymChicken":
            tune_config["env_class"] = IteratedAsymChicken
        elif hp["env_name"] in ("IteratedBoS", "IteratedAsymBoS"):
            if hp["env_name"] == "IteratedBoS":
                tune_config["env_class"] = IteratedBoS
            elif hp["env_name"] == "IteratedAsymBoS":
                tune_config["env_class"] = IteratedAsymBoS
        tune_config["num_episodes"] = (
            600000 if hp["num_episodes"] is None else hp["num_episodes"]
        )
        tune_config["trace_length"] = (
            150 if hp["trace_length"] is None else hp["trace_length"]
        )
        tune_config["batch_size"] = (
            4000 if hp["batch_size"] is None else hp["batch_size"]
        )
        tune_config["lr"] = 1.0 if hp["lr"] is None else hp["lr"]
        tune_config["gamma"] = 0.96 if hp["gamma"] is None else hp["gamma"]
        hp["x_limits"] = ((-3.0, 3.0),)
        hp["y_limits"] = ((-3.0, 3.0),)
        hp["jitter"] = 0.05
        hp["tune_class"] = LOLAPGMatrice
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "get_additional_info": True,
        }
        tune_config["metric"] = "player_row_CC_freq"

    hp["scale_multipliers"] = (
        (1 / tune_config["trace_length"], 1 / tune_config["trace_length"]),
    )
    hp["group_names"] = ["lola"]
    tune_config["seed"] = tune.grid_search(hp["lvl0_seeds"])
    tune_config["env_config"] = env_config

    stop = {"episodes_total": tune_config["num_episodes"]}

    return tune_config, stop, env_config


def train_lvl0_population(tune_hp):
    # Train with the Tune Class API (not RLLib Class)
    tune_config, stop, env_config = get_tune_config(tune_hp)
    return tune.run(
        tune_hp["tune_class"],
        name=tune_hp["exp_name"],
        config=tune_config,
        checkpoint_at_end=True,
        stop=stop,
        metric=tune_config["metric"],
        mode="max",
    )


def train_lvl1_agents(tune_hp, rllib_hp, results_list_lvl0):
    lvl0_policy_idx = 1
    lvl1_policy_idx = 0

    if tune_hp["env_name"] == "IteratedPrisonersDilemma":
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 400
        rllib_hp["base_lr"] = 0.04
        rllib_hp["x_limits"] = ((-3.5, 0.5),)
        rllib_hp["y_limits"] = ((-3.5, 0.5),)
    elif tune_hp["env_name"] == "IteratedAsymChicken":
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 400
        rllib_hp["base_lr"] = 0.04
        rllib_hp["x_limits"] = ((-11.0, 4.0),)
        rllib_hp["y_limits"] = ((-11.0, 4.0),)
    elif tune_hp["env_name"] in ("IteratedBoS", "IteratedAsymBoS"):
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 800
        rllib_hp["base_lr"] = 0.01
        rllib_hp["x_limits"] = ((-0.5, 4.5),)
        rllib_hp["y_limits"] = ((-0.5, 4.5),)
        rllib_hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 10.0),
                (
                    int(tune_hp["n_steps_per_epi"] * tune_hp["n_epi"] * 0.33),
                    2.0,
                ),
                (
                    int(tune_hp["n_steps_per_epi"] * tune_hp["n_epi"] * 0.66),
                    0.1,
                ),
            ],
            outside_value=0.1,
            framework="torch",
        )
    elif "CoinGame" in tune_hp["env_name"]:
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 4000
        rllib_hp["base_lr"] = 0.1
        rllib_hp["x_limits"] = ((-1.0, 3.0),)
        rllib_hp["y_limits"] = ((-1.0, 1.0),)
        rllib_hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (
                    int(
                        rllib_hp["n_steps_per_epi"] * rllib_hp["n_epi"] * 0.50
                    ),
                    0.1,
                ),
            ],
            outside_value=0.1,
            framework="torch",
        )
        rllib_hp["jitter"] = 0.02
    else:
        raise NotImplementedError(f'rllib_hp["env"]: {rllib_hp["env"]}')

    tune_hp.update(rllib_hp)
    stop, env_config, rllib_config = get_rllib_config(
        tune_hp, lvl1_idx=[lvl1_policy_idx], lvl1_training=True
    )

    if tune_hp["load_population"] is None:
        lvl0_checkpoints = miscellaneous.extract_checkpoints(results_list_lvl0)
    else:
        lvl0_checkpoints = tune_hp["load_population"]
    lvl0_policy_id = env_config["players_ids"][lvl0_policy_idx]
    lvl1_policy_id = env_config["players_ids"][lvl1_policy_idx]

    l1br_configuration_helper = lvl1_best_response.L1BRConfigurationHelper(
        rllib_config, lvl0_policy_id, lvl1_policy_id
    )
    l1br_configuration_helper.define_exp(
        use_n_lvl0_agents_in_each_population=len(tune_hp["lvl0_seeds"])
        // len(tune_hp["lvl1_seeds"]),
        train_n_lvl1_agents=len(tune_hp["lvl1_seeds"]),
        lvl0_checkpoints=lvl0_checkpoints,
    )
    rllib_config = l1br_configuration_helper.prepare_config_for_lvl1_training()

    results = ray.tune.run(
        DQNTrainer,
        config=rllib_config,
        stop=stop,
        name=tune_hp["exp_name"],
        checkpoint_at_end=True,
        metric="episode_reward_mean",
        mode="max",
    )

    return results


def get_rllib_config(hp: dict, lvl1_idx: list, lvl1_training: bool):
    assert lvl1_training

    tune_config, _, env_config = get_tune_config(hp=hp)
    tune_config["seed"] = 2020

    stop = {"episodes_total": hp["n_epi"]}

    after_init_fn = functools.partial(
        miscellaneous.sequence_of_fn_wt_same_args,
        function_list=[restore.after_init_load_policy_checkpoint, after_init],
    )

    def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
        return torch.optim.SGD(
            policy.q_func_vars,
            lr=policy.cur_lr,
            momentum=config["sgd_momentum"],
        )

    MyDQNTorchPolicy = DQNTorchPolicy.with_updates(
        stats_fn=log.augment_stats_fn_wt_additionnal_logs(build_q_stats),
        optimizer_fn=sgd_optimizer_dqn,
        after_init=after_init_fn,
    )

    if tune_config["env_class"] in (
        IteratedPrisonersDilemma,
        IteratedBoS,
        IteratedAsymChicken,
        IteratedAsymBoS,
    ):
        env_config.update(
            {
                "max_steps": hp["n_steps_per_epi"],
            }
        )

    elif tune_config["env_class"] in (
        VectorizedCoinGame,
        AsymVectorizedCoinGame,
    ):
        env_config.update(
            {
                "max_steps": hp["n_steps_per_epi"],
                "batch_size": 1,
            }
        )

    else:
        raise ValueError()

    tune_config["TuneTrainerClass"] = hp["tune_class"]
    tune_config["TuneTrainerClass"] = hp["tune_class"]
    tune_config["env_config"] = env_config
    policies = {}
    for policy_idx, policy_id in enumerate(env_config["players_ids"]):
        if policy_idx not in lvl1_idx:
            policies[policy_id] = (
                policy.get_tune_policy_class(DQNTorchPolicy),
                tune_config["env_class"](env_config).OBSERVATION_SPACE,
                tune_config["env_class"].ACTION_SPACE,
                {
                    "sgd_momentum": hp["sgd_momentum"],
                    "tune_config": tune_config,
                },
            )
        else:
            policies[policy_id] = (
                MyDQNTorchPolicy,
                tune_config["env_class"](env_config).OBSERVATION_SPACE,
                tune_config["env_class"].ACTION_SPACE,
                {"sgd_momentum": hp["sgd_momentum"]},
            )

    rllib_config = {
        "env": tune_config["env_class"],
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
        "min_iter_time_s": 3.0,
        # Can't restaure stuff with search
        # "seed": hp["seed"],
        "seed": tune.grid_search(
            hp["lvl1_seeds"] if lvl1_training else hp["lvl0_seeds"]
        ),
        # "evaluation_num_episodes": 100,
        # "evaluation_interval": hparams["n_epi"],
        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": hp["base_lr"],
        # Learning rate schedule
        "lr_schedule": [
            (0, hp["base_lr"]),
            (int(hp["n_steps_per_epi"] * hp["n_epi"]), hp["base_lr"] / 1e9),
        ],
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
            # The Exploration class to use. In the simplest case,
            # this is the name (str) of any class present in the
            # `rllib.utils.exploration` package.
            # You can also provide the python class directly or
            # the full location of your class (e.g.
            # "ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy").
            "type": exploration.SoftQSchedule,
            # Add constructor kwargs here (if any).
            "temperature_schedule": hp["temperature_schedule"]
            or PiecewiseSchedule(
                endpoints=[
                    (0, 10.0),
                    (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.33), 1.0),
                    (int(hp["n_steps_per_epi"] * hp["n_epi"] * 0.66), 0.1),
                ],
                outside_value=0.1,
                framework="torch",
            ),
        },
        # General config
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # LE supports only 1 worker only
        # otherwise it would be mixing several opponents trajectories
        "num_workers": 0,
        # LE supports only 1 env per worker
        # only otherwise several episodes would be played at the same time
        "num_envs_per_worker": 1,
        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and
        # `examples/custom_metrics_and_callbacks.py`
        # for more usage information.
        "callbacks": callbacks.merge_callbacks(
            log.get_logging_callbacks_class(),
            callbacks.PolicyCallbacks
            # population.PopulationOfIdenticalAlgoCallBacks
        ),
        "log_level": "INFO",
    }

    if "CoinGame" in hp["env_name"]:
        rllib_config["model"] = {
            "dim": env_config["grid_size"],
            # [Channel, [Kernel, Kernel], Stride]]
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
        }

    return stop, env_config, rllib_config


if __name__ == "__main__":
    # Only tested with the coin game environments
    debug_mode = True
    main(debug_mode)
