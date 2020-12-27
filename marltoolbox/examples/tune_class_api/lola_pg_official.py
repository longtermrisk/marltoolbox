##########
# Additional dependencies are needed:
# 1) Python 3.6
# conda install python=3.6
# 2) A fork of LOLA https://github.com/Manuscrit/lola which adds the logging through Tune
# git clone https://github.com/Manuscrit/lola
# git checkout d9c6724ea0d6bca42c8cf9688b1ff8d6fefd7267
# pip install -e .
##########

import copy
import os
import time
import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTorchPolicy

from lola.train_cg_tune_class_API import LOLAPGCG
from lola.train_pg_tune_class_API import LOLAPGMatrice

from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame
from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma, IteratedBoS
from marltoolbox.utils import policy, log, same_and_cross_perf


def get_config(tune_hparams: dict) -> dict:
    tune_config = copy.deepcopy(tune_hparams)

    # Sanity
    # assert full_config['env'] in {"CoinGame", "AsymCoinGame"}
    assert not tune_config['exact']

    # Resolve default parameters
    if tune_config['env'] in (CoinGame, AsymCoinGame):
        tune_config['num_episodes'] = 100000 if tune_config['num_episodes'] is None else tune_config['num_episodes']
        tune_config['trace_length'] = 150 if tune_config['trace_length'] is None else tune_config['trace_length']
        tune_config['batch_size'] = 4000 if tune_config['batch_size'] is None else tune_config['batch_size']
        tune_config['lr'] = 0.005 if tune_config['lr'] is None else tune_config['lr']
        tune_config['gamma'] = 0.96 if tune_config['gamma'] is None else tune_config['gamma']
        tune_hparams["x_limits"] = ((-1.0, 1.0),)
        tune_hparams["y_limits"] = ((-1.0, 1.0),)
        if tune_config['env'] == AsymCoinGame:
            tune_hparams["x_limits"] = ((-1.0, 3.0),)
        tune_hparams["jitter"] = 0.02
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "grid_size": tune_config["grid_size"],
            "get_additional_info": True,
        }
        tune_config['metric'] = "player_blue_pick_own"
    else:
        tune_config['num_episodes'] = 600000 if tune_config['num_episodes'] is None else tune_config['num_episodes']
        tune_config['trace_length'] = 150 if tune_config['trace_length'] is None else tune_config['trace_length']
        tune_config['batch_size'] = 4000 if tune_config['batch_size'] is None else tune_config['batch_size']
        tune_config['lr'] = 1.0 if tune_config['lr'] is None else tune_config['lr']
        tune_config['gamma'] = 0.96 if tune_config['gamma'] is None else tune_config['gamma']
        tune_hparams["x_limits"] = ((-3.0, 3.0),)
        tune_hparams["y_limits"] = ((-3.0, 3.0),)
        tune_hparams["jitter"] = 0.05
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "get_additional_info": True,
        }
        tune_config['metric'] = "player_row_CC"

    tune_hparams["scale_multipliers"] = ((1 / tune_config['trace_length'], 1 / tune_config['trace_length']),)
    tune_hparams["group_names"] = ["lola"]

    stop = {"episodes_total": tune_config['num_episodes']}

    return tune_config, stop, env_config


def train(tune_config, stop, tune_hp):

    print("full_config['env']", tune_config['env'])
    if tune_config['env'] in (CoinGame, AsymCoinGame):
        trainable_class = LOLAPGCG
    else:
        trainable_class = LOLAPGMatrice
    print("trainable_class", trainable_class)

    # Train with the Tune Class API (not RLLib Class)
    training_results = tune.run(trainable_class, name=tune_hp["exp_name"], config=tune_config,
                                checkpoint_at_end=True,  # checkpoint_freq=10,
                                stop=stop, metric=tune_config["metric"], mode="max")
    return training_results


def evaluate_same_and_cross_perf(training_results, rllib_config, stop, env_config, rllib_hp):

    eval_rllib_config_update = {
        "env": rllib_hp["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env"].ACTION_SPACE,
                    {"tune_config": rllib_config}),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env"].ACTION_SPACE,
                    {"tune_config": rllib_config}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": rllib_hp["seed"],
        "model": {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],  # [Channel, [Kernel, Kernel], Stride]]
        },
        "min_iter_time_s": 3.0,
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])

    trainable_class = LOLAPGCG if rllib_hp['env'] in (CoinGame, AsymCoinGame) else LOLAPGMatrice

    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluation(TuneTrainerClass=trainable_class,
                                                               group_names=rllib_hp["group_names"],
                                                               evaluation_config=eval_rllib_config_update,
                                                               stop_config=stop,
                                                               exp_name=rllib_hp["exp_name"],
                                                               policies_to_train=["None"],
                                                               policies_to_load_from_checkpoint=policies_to_load,
                                                               )

    if rllib_hp["load_plot_data"] is None:
        analysis_metrics_per_mode = evaluator.perf_analysis(n_same_play_per_checkpoint=1,
                                                            n_cross_play_per_checkpoint=min(5, train_n_replicates-1),
                                                            extract_checkpoints_from_results=[training_results],
                                                            )
    else:
        analysis_metrics_per_mode = evaluator.load_results(to_load_path=rllib_hp["load_plot_data"])

    evaluator.plot_results(analysis_metrics_per_mode,
                           title_sufix=": " + rllib_hp['env'].NAME,
                           metrics=((f"policy_reward_mean/{env_config['players_ids'][0]}",
                                     f"policy_reward_mean/{env_config['players_ids'][1]}"),),
                           x_limits=rllib_hp["x_limits"], y_limits=rllib_hp["y_limits"],
                           scale_multipliers=rllib_hp["scale_multipliers"],
                           markersize=5,
                           alpha=1.0,
                           jitter=rllib_hp["jitter"],
                           colors=["red", "blue"],
                           xlabel="player 1 payoffs", ylabel="player 2 payoffs", add_title=False, frameon=True,
                           show_groups=False,  plot_max_n_points=train_n_replicates
                           )



if __name__ == "__main__":
    debug = False
    train_n_replicates = 4 if debug else 40
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("LOLA_PG")

    tune_hparams = {
        "exp_name": exp_name,


        # Print metrics
        "load_plot_data": None,
        # CG 40 seeds
        "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-60-cpu-1-preemtible/LOLA_PG/2020_12_21/07_11_03/2020_12_21/09_53_39/SameAndCrossPlay_save.p",
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-60-cpu-1-preemtible/LOLA_PG/2020_12_21/10_20_34/2020_12_21/13_03_51/SameAndCrossPlay_save.p",
        # ACG 40 seeds
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-20-cpu-2-memory-x2/LOLA_PG/2020_12_21/06_56_48/2020_12_21/12_18_50/SameAndCrossPlay_save.p",
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-60-cpu-1-preemtible/LOLA_PG/2020_12_21/10_20_34/2020_12_21/13_03_51/SameAndCrossPlay_save.p",

        # CG
        # Population of 5
        # "load_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-3/LOLA_PG/2020_12_15/20_55_11/lvl1_results.p",
        # Population of 40
        # "load_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-20-cpu-1-memory-x2/L1BR_LOLA_PG/2020_12_18/09_00_30/lvl1_results.p",
        # ACG
        # "load_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-3/LOLA_PG/2020_12_15/20_56_24/lvl1_results.p",
        # Population of 40
        # "load_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-20-cpu-2-memory-x2/L1BR_LOLA_PG/2020_12_18"
        #              "/09_03_13/lvl1_results.p",

        # Dynamically set
        "num_episodes": 5 if debug else 2000,
        "trace_length": 5 if debug else 20,
        # "trace_length": 5 if debug else None,
        # "trace_length": tune.grid_search([150, 75]),
        "lr": None,
        # "lr": 0.005 / 10,  # None,
        # "gamma": 0.5 if debug else None,
        "gamma": 0.5,
        # "gamma": tune.grid_search([0.5, 0.96]),
        # !!! To use the default batch size with coin game, you need 35Go of memory per seed run in parallel !!!
        # "batch_size": None, # To use the defaults values from the official repository.
        "batch_size": 5 if debug else 512,
        # "batch_size": 20 if debug else None, #1024,
        # "batch_size": tune.grid_search([512, 256]),

        # "env": IteratedPrisonersDilemma,
        # "env": IteratedBoS,
        # "env": IteratedAsymBoS,
        "env": CoinGame,
        # "env": AsymCoinGame,

        "pseudo": False,
        "grid_size": 3,
        "lola_update": True,
        "opp_model": False,
        "mem_efficient": True,
        "lr_correction": 1,
        # "bs_mul": 1,
        "bs_mul": 1 / 10,
        # "bs_mul": tune.grid_search([1/10, 1/30]),
        "simple_net": True,
        "hidden": 32,
        "reg": 0,
        "set_zero": 0,

        # "exact": True,
        "exact": False,

        "warmup": 1,  # False,

        "seed": tune.grid_search(seeds),

        "changed_config": False,
        "ac_lr": 1.0,
        # "ac_lr": 0.005,
        "summary_len": 1,
        "use_MAE": False,
        # "use_MAE": True,

        # "use_toolbox_env": False,
        "use_toolbox_env": True,

        "clip_loss_norm": False,
        # "clip_loss_norm": 10.0,
        "clip_lola_update_norm": False,
        # "clip_lola_update_norm": 0.5,
        "clip_lola_correction_norm": 3.0,
        # "clip_lola_correction_norm": tune.grid_search([10.0, 3.0]),
        "clip_lola_actor_norm": 10.0,
        # "clip_lola_actor_norm": tune.grid_search([10.0, 3.0]),

        "entropy_coeff": 0.001,
        # "entropy_coeff": tune.grid_search([0.0, 0.0003, 0.001, 0.003, 0.01]),
        # "entropy_coeff": tune.grid_search(s[0.0, 0.001, 0.01]),

        # "weigth_decay": 0.0,  # 0.001 working well
        "weigth_decay": 0.03,  # 0.001 working well
        # "weigth_decay": tune.grid_search([0.03, 0.1]),  # 0.001 working well

        "lola_correction_multiplier": 1,
        # "lola_correction_multiplier": tune.grid_search([1, 1/3, 1/10]),

        "lr_decay": True,

        "correction_reward_baseline_per_step": False,
        # "correction_reward_baseline_per_step": tune.grid_search([False, True]),

        "use_critic": False,
        # "use_critic": tune.grid_search([False, True]),

    }

    if tune_hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0)

        full_config, stop, env_config = get_config(tune_hparams)
        training_results = train(full_config, stop, tune_hp=tune_hparams)

        rllib_hparams = copy.deepcopy(tune_hparams)
        rllib_hparams["seed"] = 2020
        # rllib_hparams["batch_size"] = 1
        rllib_hparams["num_episodes"] = 100
        eval_tune_config, stop, env_config = get_config(rllib_hparams)
        env_config["batch_size"] = 1
        eval_tune_config['TuneTrainerClass'] = LOLAPGCG
        evaluate_same_and_cross_perf(training_results, eval_tune_config, stop, env_config, rllib_hparams)

        ray.shutdown()
    else:
        rllib_hparams = copy.deepcopy(tune_hparams)
        rllib_hparams["seed"] = 2020
        # rllib_hparams["batch_size"] = 1
        rllib_hparams["num_episodes"] = 100
        eval_tune_config, stop, env_config = get_config(rllib_hparams)
        env_config["batch_size"] = 1
        eval_tune_config['TuneTrainerClass'] = LOLAPGCG
        evaluate_same_and_cross_perf(None, eval_tune_config, stop, env_config, rllib_hparams)

        ray.shutdown()
