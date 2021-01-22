##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the tune_class_api/lola_pg_official.py file
##########

import copy

import functools
import os
import ray
import time
import torch
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, build_q_stats, after_init
from ray.rllib.utils.schedules import PiecewiseSchedule

from marltoolbox.algos.lola.train_cg_tune_class_API import LOLAPGCG
from marltoolbox.algos.lola.train_pg_tune_class_API import LOLAPGMatrice
from marltoolbox.algos import population
from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame
from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma, IteratedBoS, IteratedAsymChicken, IteratedAsymBoS
from marltoolbox.utils import policy, log, miscellaneous, lvl1_best_response, exploration, restore


# TODO make it work for all env (not only ACG and CG)? or only for them

def get_tune_config(hp: dict) -> dict:
    tune_config = copy.deepcopy(hp)
    assert not tune_config['exact']

    # Resolve default parameters
    if tune_config['env'] in (CoinGame, AsymCoinGame):
        tune_config['num_episodes'] = 100000 if tune_config['num_episodes'] is None else tune_config['num_episodes']
        tune_config['trace_length'] = 150 if tune_config['trace_length'] is None else tune_config['trace_length']
        tune_config['batch_size'] = 4000 if tune_config['batch_size'] is None else tune_config['batch_size']
        tune_config['lr'] = 0.005 if tune_config['lr'] is None else tune_config['lr']
        tune_config['gamma'] = 0.96 if tune_config['gamma'] is None else tune_config['gamma']
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
        }
        tune_config['metric'] = "player_blue_pick_speed"
    else:
        tune_config['num_episodes'] = 600000 if tune_config['num_episodes'] is None else tune_config['num_episodes']
        tune_config['trace_length'] = 150 if tune_config['trace_length'] is None else tune_config['trace_length']
        tune_config['batch_size'] = 4000 if tune_config['batch_size'] is None else tune_config['batch_size']
        tune_config['lr'] = 1.0 if tune_config['lr'] is None else tune_config['lr']
        tune_config['gamma'] = 0.96 if tune_config['gamma'] is None else tune_config['gamma']
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
        tune_config['metric'] = "player_row_CC"

    hp["scale_multipliers"] = ((1 / tune_config['trace_length'], 1 / tune_config['trace_length']),)
    hp["group_names"] = ["lola"]
    tune_config["seed"] = tune.grid_search(hp["lvl0_seeds"])

    stop = {"episodes_total": tune_config['num_episodes']}

    return tune_config, stop, env_config


def get_rllib_config(hp: dict, lvl1_idx: list, lvl1_training: bool):
    assert lvl1_training

    tune_config, _, _ = get_tune_config(hp=hp)
    tune_config["seed"] = 2020

    stop = {"episodes_total": hp['n_epi']}

    after_init_fn = functools.partial(
        miscellaneous.sequence_of_fn_wt_same_args,
        function_list=[restore.after_init_load_policy_checkpoint, after_init]
    )

    def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
        return torch.optim.SGD(policy.q_func_vars, lr=policy.cur_lr, momentum=config["sgd_momentum"])

    MyDQNTorchPolicy = DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats),
                                                   optimizer_fn=sgd_optimizer_dqn,
                                                   after_init=after_init_fn)

    if hp["env"] in (IteratedPrisonersDilemma, IteratedBoS, IteratedAsymChicken, IteratedAsymBoS):
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hp["n_steps_per_epi"],
            "reward_randomness": 0.1 if hp["env"] == IteratedAsymChicken else 0.0,
            "get_additional_info": True,
        }

    elif hp["env"] in (CoinGame, AsymCoinGame):
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hp["n_steps_per_epi"],
            "batch_size": 1,
            "grid_size": 3,
            "get_additional_info": True,
        }

    else:
        raise ValueError

    tune_config["TuneTrainerClass"] = hp["tune_class"]
    tune_config["TuneTrainerClass"] = hp["tune_class"]
    policies = {}
    for policy_idx, policy_id in enumerate(env_config["players_ids"]):
        if policy_idx not in lvl1_idx:
            policies[policy_id] = (
                policy.get_tune_policy_class(DQNTorchPolicy),
                hp["env"](env_config).OBSERVATION_SPACE,
                hp["env"].ACTION_SPACE,
                {"sgd_momentum": hp["sgd_momentum"],
                 "tune_config": tune_config}
            )
        else:
            policies[policy_id] = (
                MyDQNTorchPolicy,
                hp["env"](env_config).OBSERVATION_SPACE,
                hp["env"].ACTION_SPACE,
                {"sgd_momentum": hp["sgd_momentum"]}
            )

    rllib_config = {
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
        "min_iter_time_s": 3.0,
        # Can't restaure stuff with search
        # "seed": hp["seed"],
        "seed": tune.grid_search(hp["lvl1_seeds"] if lvl1_training else hp["lvl0_seeds"]),

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
        "callbacks": miscellaneous.merge_callbacks(log.get_logging_callbacks_class(),
                                                   population.PopulationOfIdenticalAlgoCallBacks),

        "log_level": "INFO",

    }

    if hp["env"] in (CoinGame, AsymCoinGame):
        rllib_config["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],  # [Channel, [Kernel, Kernel], Stride]]
        }

    return stop, env_config, rllib_config


def train_lvl0_population(tune_hp):
    # Train with the Tune Class API (not RLLib Class)
    tune_config, stop, env_config = get_tune_config(tune_hp)
    return tune.run(tune_hp["tune_class"], name=tune_hp["exp_name"], config=tune_config,
                    checkpoint_at_end=True, stop=stop, metric=tune_config["metric"], mode="max")


def train_lvl1_agents(tune_hp, rllib_hp, results_list_lvl0):
    lvl0_policy_idx = 1
    lvl1_policy_idx = 0

    if tune_hp["env"] == IteratedPrisonersDilemma:
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 400
        rllib_hp["base_lr"] = 0.04
        rllib_hp["x_limits"] = ((-3.5, 0.5),)
        rllib_hp["y_limits"] = ((-3.5, 0.5),)
    elif tune_hp["env"] == IteratedAsymChicken:
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 400
        rllib_hp["base_lr"] = 0.04
        rllib_hp["x_limits"] = ((-11.0, 4.0),)
        rllib_hp["y_limits"] = ((-11.0, 4.0),)
    elif tune_hp["env"] in (IteratedBoS, IteratedAsymBoS):
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 800
        rllib_hp["base_lr"] = 0.01
        rllib_hp["x_limits"] = ((-0.5, 4.5),)
        rllib_hp["y_limits"] = ((-0.5, 4.5),)
        rllib_hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 10.0),
                (int(tune_hp["n_steps_per_epi"] * tune_hp["n_epi"] * 0.33), 2.0),
                (int(tune_hp["n_steps_per_epi"] * tune_hp["n_epi"] * 0.66), 0.1)],
            outside_value=0.1,
            framework="torch")
    elif tune_hp["env"] in [CoinGame, AsymCoinGame]:
        rllib_hp["n_epi"] = 3 if rllib_hp["debug"] else 4000
        rllib_hp["base_lr"] = 0.1
        rllib_hp["x_limits"] = ((-1.0, 3.0),)
        rllib_hp["y_limits"] = ((-1.0, 1.0),)
        rllib_hp["temperature_schedule"] = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (int(rllib_hp["n_steps_per_epi"] * rllib_hp["n_epi"] * 0.50), 0.1)],
            outside_value=0.1,
            framework="torch")
        rllib_hp["jitter"] = 0.02
    else:
        raise NotImplementedError(f'rllib_hp["env"]: {rllib_hp["env"]}')

    tune_hp.update(rllib_hp)
    stop, env_config, rllib_config = get_rllib_config(tune_hp, lvl1_idx=[lvl1_policy_idx], lvl1_training=True)

    if tune_hp["load_population"] is None:
        lvl0_checkpoints = miscellaneous.extract_checkpoints(results_list_lvl0)
    else:
        lvl0_checkpoints = tune_hp["load_population"]
    lvl0_policy_id = env_config["players_ids"][lvl0_policy_idx]
    lvl1_policy_id = env_config["players_ids"][lvl1_policy_idx]

    lvl1_best_response.prepare_config_for_lvl1_training(
        config=rllib_config,
        lvl0_policy_id=lvl0_policy_id, lvl1_policy_id=lvl1_policy_id,
        select_n_lvl0_from_population=len(tune_hp["lvl0_seeds"]) // len(tune_hp["lvl1_seeds"]),
        n_lvl1_to_train=len(tune_hp["lvl1_seeds"]),
        overlapping_population=False, lvl0_checkpoints=lvl0_checkpoints)

    print("rllib_config", rllib_config)
    results = ray.tune.run(DQNTrainer, config=rllib_config,
                           stop=stop, name=tune_hp["exp_name"],
                           checkpoint_at_end=True,
                           metric="episode_reward_mean", mode="max")

    return results


def main(debug):
    n_in_lvl0_population = 2 if debug else 40
    n_lvl1 = 1 if debug else 1
    timestamp = int(time.time())
    lvl0_seeds = [seed + timestamp for seed in list(range(n_in_lvl0_population))]
    lvl1_seeds = list(range(n_lvl1))

    exp_name, _ = log.log_in_current_day_dir("L1BR_LOLA_PG")

    tune_hparams = {
        "exp_name": exp_name,

        "load_data": None,
        # Example: "load_data": ".../lvl1_results.p",
        "load_population": None,
        # Example: "load_population": [".../checkpoint.json", ".../checkpoint.json", ...]

        "num_episodes": 5 if debug else 2000,
        "trace_length": 5 if debug else 20,
        "lr": None,
        "gamma": 0.5,
        "batch_size": 5 if debug else 512,

        # "env": IteratedPrisonersDilemma,
        # "env": IteratedBoS,
        # "env": IteratedAsymBoS,
        # "env": CoinGame,
        "env": AsymCoinGame,

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

        results_list_lvl1 = train_lvl1_agents(tune_hp=tune_hparams, rllib_hp=rllib_hparams,
                                              results_list_lvl0=results_list_lvl0)
        log.save_metrics(results_list_lvl1, exp_name, "lvl1_results.p", limit=True)

        ray.shutdown()
    else:
        log.pprint_saved_metrics(tune_hparams["load_data"],
                                 keywords_to_print=["policy_reward_mean", "speed.*mean", "own.*mean", "analysis",
                                                    "^avg$", "last-10-avg"])

if __name__ == "__main__":
    # Only tested with the con game environments
    debug_mode = True
    main(debug_mode)
