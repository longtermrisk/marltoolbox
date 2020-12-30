##########
# Additional dependencies are needed:
# 1) Python 3.6
# conda install python=3.6
# 2) A fork of LOLA https://github.com/Manuscrit/lola which adds the logging through Tune
# git clone https://github.com/Manuscrit/lola
# git checkout 181cb6dfa0ebf85807d42f1f770b0556a8f4f4d6
# pip install -e .
##########

import copy

import os
import ray
import time
from ray import tune
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy

from lola.train_exact_tune_class_API import LOLAExact
from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma, IteratedMatchingPennies, IteratedAsymBoS
from marltoolbox.utils import policy, log, same_and_cross_perf


def get_tune_config(hp: dict) -> dict:
    config = copy.deepcopy(hp)
    assert config['env'] in ("IPD", "IMP", "BoS", "AsymBoS")

    if config["env"] in ("IPD", "IMP", "BoS", "AsymBoS"):
        # config["make_policy"] = ("make_simple_policy", {})

        env_config = {
            "players_ids": ["player_row", "player_col"],
            # "batch_size": config["batch_size"],
            "max_steps": config["trace_length"],
            "get_additional_info": True,
        }
        # config['metric'] = "ag_0_returns_player_1"

    if config["env"] in ("IPD", "BoS", "AsymBoS"):
        config["gamma"] = 0.96 if config["gamma"] is None else config["gamma"]
        config["save_dir"] = "dice_results_ipd"
    elif config["env"] == "IMP":
        config["gamma"] = 0.9 if config["gamma"] is None else config["gamma"]
        config["save_dir"] = "dice_results_imp"

    stop = {"episodes_total": config['num_episodes']}

    return config, stop, env_config


def train(hp):
    tune_config, stop, _ = get_tune_config(hp)
    # Train with the Tune Class API (not RLLib Class)
    training_results = tune.run(LOLAExact, name=hp["exp_name"], config=tune_config,
                                checkpoint_at_end=True,
                                stop=stop, metric=hp["metric"], mode="max")
    return training_results


def evaluate(training_results, hp):
    hp_eval = copy.deepcopy(hp)

    plot_config = {}

    hp_eval["min_iter_time_s"] = 3.0
    hp_eval["seed"] = 2020
    hp_eval["batch_size"] = 1
    hp_eval["num_episodes"] = 100
    tune_config, stop, env_config = get_tune_config(hp_eval)
    tune_config['TuneTrainerClass'] = LOLAExact

    plot_config["group_names"] = ["lola"]
    plot_config["scale_multipliers"] = ((1 / tune_config['trace_length'], 1 / tune_config['trace_length']),)
    plot_config["jitter"] = 0.05

    if hp_eval["env"] == "IPD":
        hp_eval["env"] = IteratedPrisonersDilemma
        plot_config["x_limits"] = ((-3.5, 0.5),)
        plot_config["y_limits"] = ((-3.5, 0.5),)
    elif hp_eval["env"] == "IMP":
        hp_eval["env"] = IteratedMatchingPennies
        plot_config["x_limits"] = ((-1.0, 1.0),)
        plot_config["y_limits"] = ((-1.0, 1.0),)
    # elif hp_eval["env"] == "BoS":
    #     hp_eval["env"] = IteratedBoS
    #     plot_config["x_limits"] = ((-0.5, 3.5),)
    #     plot_config["y_limits"] = ((-0.5, 3.5),)
    elif hp_eval["env"] == "AsymBoS":
        hp_eval["env"] = IteratedAsymBoS
        plot_config["x_limits"] = ((-1.0, 5.0),)
        plot_config["y_limits"] = ((-1.0, 5.0),)
    # elif hp_eval["env"] == "CoinGame":
    #     hp_eval["env"] = CoinGame
    #     plot_config["x_limits"] = ((-1.0, 3.0),)
    #     plot_config["y_limits"] = ((-1.0, 3.0),)
    #     plot_config["jitter"] = 0.02
    # elif hp_eval["env"] == "AsymCoinGame":
    #     hp_eval["env"] = AsymCoinGame
    #     plot_config["x_limits"] = ((-1.0, 3.0),)
    #     plot_config["y_limits"] = ((-1.0, 3.0),)
    #     plot_config["jitter"] = 0.02
    else:
        raise NotImplementedError()

    rllib_config = {
        "env": hp_eval["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    hp_eval["env"](env_config).OBSERVATION_SPACE,
                    hp_eval["env"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)}),

                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    hp_eval["env"](env_config).OBSERVATION_SPACE,
                    hp_eval["env"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": hp_eval["seed"],
        "min_iter_time_s": hp_eval["min_iter_time_s"],
    }

    evaluate_same_and_cross_perf(training_results, rllib_config, stop, env_config, hp_eval, plot_config)


def evaluate_same_and_cross_perf(training_results, rllib_config, stop, env_config, hp, plot_config):
    policies_to_load = copy.deepcopy(env_config["players_ids"])

    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluation(TuneTrainerClass=LOLAExact,
                                                               group_names=plot_config["group_names"],
                                                               evaluation_config=rllib_config,
                                                               stop_config=stop,
                                                               exp_name=hp["exp_name"],
                                                               policies_to_train=["None"],
                                                               policies_to_load_from_checkpoint=policies_to_load,
                                                               )

    if hparams["load_plot_data"] is None:
        analysis_metrics_per_mode = evaluator.perf_analysis(n_same_play_per_checkpoint=1,
                                                            n_cross_play_per_checkpoint=1,
                                                            extract_checkpoints_from_results=[training_results],
                                                            )
    else:
        analysis_metrics_per_mode = evaluator.load_results(to_load_path=hparams["load_plot_data"])

    evaluator.plot_results(analysis_metrics_per_mode,
                           title_sufix=": " + hp['env'].NAME,
                           metrics=((f"policy_reward_mean/{env_config['players_ids'][0]}",
                                     f"policy_reward_mean/{env_config['players_ids'][1]}"),),
                           x_limits=plot_config["x_limits"], y_limits=plot_config["y_limits"],
                           scale_multipliers=plot_config["scale_multipliers"],
                           markersize=5,
                           alpha=1.0,
                           jitter=plot_config["jitter"],
                           colors=["red", "blue"],
                           xlabel="player 1 payoffs", ylabel="player 2 payoffs", add_title=False, frameon=True,
                           show_groups=False, plot_max_n_points=train_n_replicates
                           )


if __name__ == "__main__":
    debug = False
    train_n_replicates = 2 if debug else 40
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("LOLA_Exact")

    hparams = {

        "load_plot_data": None,
        # IPD
        # 5 seeds
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-2/LOLA_DICE/2020_12_15/11_44_21/2020_12_15/12_07_42/SameAndCrossPlay_save.p",
        # 40 seeds
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-4/LOLA_DICE/21_10_16/2020_12_16/22_52_00/SameAndCrossPlay_save.p",
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-3/LOLA_Exact/2020_12_18/14_04_07/2020_12_18/14_08_27/SameAndCrossPlay_save.p",
        # BOS
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-3/LOLA_DICE/2020_12_15/12_04_40/2020_12_15/12_28_37/SameAndCrossPlay_save.p",
        # ABoS
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-3/LOLA_Exact/2020_12_18/17_45_15/2020_12_18/17_50_02/SameAndCrossPlay_save.p",
        # ABoS 40 seeds
        # "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-3/LOLA_Exact/2020_12_18/17_45_15/2020_12_18/17_50_02/SameAndCrossPlay_save.p",
        # ABos 40 seeds true payoff matrix
        "load_plot_data": "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-2/LOLA_Exact/2020_12_22/09_10_23/2020_12_22/09_15_57/SameAndCrossPlay_save.p",

        "exp_name": exp_name,

        # "env": "IPD",
        # "env": "IMP",
        "env": "AsymBoS",

        "num_episodes": 5 if debug else 50,
        "trace_length": 200,
        "simple_net": True,
        "corrections": True,
        "pseudo": False,
        "num_hidden": 10,
        "reg": 0.0,
        "lr": 1.,
        "lr_correction": 0.5,
        "gamma": 0.96,

        "seed": tune.grid_search(seeds),
        "metric": "ret1",
    }

    if hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0)
        training_results = train(hparams)
    else:
        training_results = None

    evaluate(training_results, hparams)
    ray.shutdown()
