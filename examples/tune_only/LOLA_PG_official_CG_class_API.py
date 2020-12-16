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


def get_config(hparams: dict) -> dict:
    full_config = copy.deepcopy(hparams)

    # Sanity
    # assert full_config['env'] in {"CoinGame", "AsymCoinGame"}
    assert not full_config['exact']

    # Resolve default parameters
    if full_config['env'] in (CoinGame, AsymCoinGame):
        full_config['num_episodes'] = 100000 if full_config['num_episodes'] is None else full_config['num_episodes']
        full_config['trace_length'] = 150 if full_config['trace_length'] is None else full_config['trace_length']
        full_config['batch_size'] = 4000 if full_config['batch_size'] is None else full_config['batch_size']
        full_config['lr'] = 0.005 if full_config['lr'] is None else full_config['lr']
        full_config['gamma'] = 0.96 if full_config['gamma'] is None else full_config['gamma']
        hparams["x_limits"] = ((-1.0, 1.0),)
        hparams["y_limits"] = ((-1.0, 1.0),)
        if full_config['env'] == AsymCoinGame:
            hparams["x_limits"] = ((-1.0, 3.0),)
        hparams["jitter"] = 0.02
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "batch_size": full_config["batch_size"],
            "max_steps": full_config["trace_length"],
            "grid_size": full_config["grid_size"],
            "get_additional_info": True,
        }
        full_config['metric'] = "player_blue_pick_own"
    else:
        full_config['num_episodes'] = 600000 if full_config['num_episodes'] is None else full_config['num_episodes']
        full_config['trace_length'] = 150 if full_config['trace_length'] is None else full_config['trace_length']
        full_config['batch_size'] = 4000 if full_config['batch_size'] is None else full_config['batch_size']
        full_config['lr'] = 1.0 if full_config['lr'] is None else full_config['lr']
        full_config['gamma'] = 0.96 if full_config['gamma'] is None else full_config['gamma']
        hparams["x_limits"] = ((-3.0, 3.0),)
        hparams["y_limits"] = ((-3.0, 3.0),)
        hparams["jitter"] = 0.05
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "batch_size": full_config["batch_size"],
            "max_steps": full_config["trace_length"],
            "get_additional_info": True,
        }
        full_config['metric'] = "player_row_CC"

    hparams["scale_multipliers"] = ((1 / full_config['trace_length'],1 / full_config['trace_length']),)
    hparams["group_names"] = ["lola"]

    stop = {"episodes_total": full_config['num_episodes']}

    return full_config, stop, env_config


def train(full_config, stop, hp):

    print("full_config['env']", full_config['env'])
    if full_config['env'] in (CoinGame, AsymCoinGame):
        trainable_class = LOLAPGCG
    else:
        trainable_class = LOLAPGMatrice
    print("trainable_class", trainable_class)

    # Train with the Tune Class API (not RLLib Class)
    training_results = tune.run(trainable_class, name=hp["exp_name"], config=full_config,
                                checkpoint_at_end=True,  # checkpoint_freq=10,
                                stop=stop, metric=full_config["metric"], mode="max")
    return training_results


def evaluate_same_and_cross_perf(training_results, eval_tune_config, stop, env_config, hp):

    eval_rllib_config_update = {
        "env": hp["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    hp["env"](env_config).OBSERVATION_SPACE,
                    hp["env"].ACTION_SPACE,
                    copy.deepcopy(eval_tune_config)),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    hp["env"](env_config).OBSERVATION_SPACE,
                    hp["env"].ACTION_SPACE,
                    copy.deepcopy(eval_tune_config)),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": hp["seed"],
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])

    trainable_class = LOLAPGCG if hp['env'] in (CoinGame, AsymCoinGame) else LOLAPGMatrice

    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluation(TuneTrainerClass=trainable_class,
                                                               group_names=hp["group_names"],
                                                               evaluation_config=eval_rllib_config_update,
                                                               stop_config=stop,
                                                               exp_name=hp["exp_name"],
                                                               policies_to_train=["None"],
                                                               policies_to_load_from_checkpoint=policies_to_load,
                                                               )

    if hparams["load_plot_data"] is None:
        analysis_metrics_per_mode = evaluator.perf_analysis(n_same_play_per_checkpoint=1,
                                                            n_cross_play_per_checkpoint=(train_n_replicates * len(
                                                                hp["group_names"])) - 1,
                                                            extract_checkpoints_from_results=[training_results],
                                                            )
    else:
        analysis_metrics_per_mode = evaluator.load_results(to_load_path=hparams["load_plot_data"])

    evaluator.plot_results(analysis_metrics_per_mode,
                           title_sufix=": " + hp['env'].NAME,
                           metrics=((f"policy_reward_mean/{env_config['players_ids'][0]}",
                                     f"policy_reward_mean/{env_config['players_ids'][1]}"),),
                           x_limits=hp["x_limits"], y_limits=hp["y_limits"],
                           scale_multipliers=hp["scale_multipliers"],
                           markersize=5,
                           alpha=1.0,
                           jitter=hp["jitter"],
                           colors=["red", "blue"],
                           xlabel="player 1 payoffs", ylabel="player 2 payoffs", add_title=False, frameon=True,
                           show_groups=False
                           )



if __name__ == "__main__":
    debug = False
    train_n_replicates = 5
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.put_everything_in_one_dir("LOLA_PG")

    hparams = {
        "exp_name": exp_name,

        "load_plot_data": None,
        # IPD
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-1/2020_12_09/20_47_26/2020_12_09/21_00_14/SameAndCrossPlay_save.p",
        # BOS
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-1/2020_12_09/20_47_34/2020_12_09/21_02_25/SameAndCrossPlay_save.p",
        # CG
        "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-6-memory-x2/LOLA_PG/2020_12_15/20_40_25/2020_12_15/23_38_51/SameAndCrossPlay_save.p",
        # ACG
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-6-memory-x2/LOLA_PG/2020_12_15"
        #                   "/20_40_55/2020_12_15/23_41_57/SameAndCrossPlay_save.p",

        # Dynamically set
        "num_episodes": 5 if debug else 2000,
        "trace_length": 20,
        # "trace_length": 5 if debug else None,
        # "trace_length": tune.grid_search([150, 75]),
        "lr": None,
        # "lr": 0.005 / 10,  # None,
        # "gamma": 0.5 if debug else None,
        "gamma": 0.5,
        # "gamma": tune.grid_search([0.5, 0.96]),
        # !!! To use the default batch size with coin game, you need 35Go of memory per seed run in parallel !!!
        # "batch_size": None, # To use the defaults values from the official repository.
        "batch_size": 512,
        # "batch_size": 20 if debug else None, #1024,
        # "batch_size": tune.grid_search([512, 256]),

        # "env": IteratedPrisonersDilemma,
        # "env": IteratedBoS,
        "env": CoinGame,
        # "env": AsymCoinGame,

        "pseudo": False,
        "grid_size": 3,
        "lola_update": True,
        "opp_model": False,
        "mem_efficient": True,
        "lr_correction": 1,
        # "bs_mul": 1,
        "bs_mul": 1/10,
        # "bs_mul": tune.grid_search([1/10, 1/30, 1/60, 1/90]),
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
        # "entropy_coeff": tune.grid_search([0.0, 0.001, 0.01]),

        # "weigth_decay": 0.0,  # 0.001 working well
        "weigth_decay": 0.1,  # 0.001 working well
        # "weigth_decay": tune.grid_search([0.03, 0.1]),  # 0.001 working well

        "lola_correction_multiplier": 1,
        # "lola_correction_multiplier": tune.grid_search([1, 1/3, 1/10]),

        "lr_decay": True,

        "correction_reward_baseline_per_step": False,
        # "correction_reward_baseline_per_step": tune.grid_search([False, True]),

        "use_critic": False,
        # "use_critic": tune.grid_search([False, True]),

    }

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0)

        full_config, stop, env_config = get_config(hparams)
        training_results = train(full_config, stop, hp=hparams)

        hparams_eval = copy.deepcopy(hparams)
        hparams_eval["seed"] = 2020
        hparams_eval["batch_size"] = 1
        hparams_eval["num_episodes"] = 100
        eval_tune_config, stop, env_config = get_config(hparams_eval)
        eval_tune_config['TuneTrainerClass'] = LOLAPGCG
        evaluate_same_and_cross_perf(training_results, eval_tune_config, stop, env_config, hparams_eval)

        ray.shutdown()
    else:
        hparams_eval = copy.deepcopy(hparams)
        hparams_eval["seed"] = 2020
        hparams_eval["batch_size"] = 1
        hparams_eval["num_episodes"] = 100
        eval_tune_config, stop, env_config = get_config(hparams_eval)
        eval_tune_config['TuneTrainerClass'] = LOLAPGCG
        evaluate_same_and_cross_perf(None, eval_tune_config, stop, env_config, hparams_eval)

        ray.shutdown()
