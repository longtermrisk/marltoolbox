##########
# Additional dependencies are needed:
# 1) Python 3.6
# conda install python=3.6
# reinstall marltoolbox and RLLib with Python 3.6 (see README.md)
# 2) A fork of LOLA https://github.com/Manuscrit/lola
# git clone https://github.com/Manuscrit/lola
# git checkout 181cb6dfa0ebf85807d42f1f770b0556a8f4f4d6
# cd lola
# pip install -e .
##########

import copy
import os
import time

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTorchPolicy

from marltoolbox.algos.lola.train_cg_tune_class_API import LOLAPGCG
from marltoolbox.algos.lola.train_pg_tune_class_API import LOLAPGMatrice
from marltoolbox.envs.vectorized_coin_game import CoinGame, AsymCoinGame
from marltoolbox.utils import policy, log, same_and_cross_perf
from marltoolbox.utils.plot import PlotConfig


def get_tune_config(tune_hp: dict, stop_on_epi_number: bool = False):
    tune_config = copy.deepcopy(tune_hp)

    assert not tune_config['exact']

    # Resolve default parameters
    if tune_config['env'] in (CoinGame, AsymCoinGame):
        tune_config['num_episodes'] = 100000 if tune_config['num_episodes'] is None else tune_config['num_episodes']
        tune_config['trace_length'] = 150 if tune_config['trace_length'] is None else tune_config['trace_length']
        tune_config['batch_size'] = 4000 if tune_config['batch_size'] is None else tune_config['batch_size']
        tune_config['lr'] = 0.005 if tune_config['lr'] is None else tune_config['lr']
        tune_config['gamma'] = 0.96 if tune_config['gamma'] is None else tune_config['gamma']
        tune_hp["x_limits"] = (-1.0, 1.0)
        tune_hp["y_limits"] = (-1.0, 1.0)
        if tune_config['env'] == AsymCoinGame:
            tune_hp["x_limits"] = (-1.0, 3.0)
        tune_hp["jitter"] = 0.02
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "grid_size": tune_config["grid_size"],
            "get_additional_info": True,
        }
        tune_config['metric'] = "player_blue_pick_own_color"
    else:
        tune_config['num_episodes'] = 600000 if tune_config['num_episodes'] is None else tune_config['num_episodes']
        tune_config['trace_length'] = 150 if tune_config['trace_length'] is None else tune_config['trace_length']
        tune_config['batch_size'] = 4000 if tune_config['batch_size'] is None else tune_config['batch_size']
        tune_config['lr'] = 1.0 if tune_config['lr'] is None else tune_config['lr']
        tune_config['gamma'] = 0.96 if tune_config['gamma'] is None else tune_config['gamma']
        tune_hp["x_limits"] = (-3.0, 3.0)
        tune_hp["y_limits"] = (-3.0, 3.0)
        tune_hp["jitter"] = 0.05
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "get_additional_info": True,
        }
        tune_config['metric'] = "player_row_CC"

    tune_hp["scale_multipliers"] = (1 / tune_config['trace_length'], 1 / tune_config['trace_length'])

    if stop_on_epi_number:
        stop = {"episodes_total": tune_config['num_episodes']}
    else:
        stop = {"finished": True}

    return tune_config, stop, env_config


def train(tune_hp):
    tune_config, stop, env_config = get_tune_config(tune_hp)

    print("full_config['env']", tune_config['env'])
    if tune_config['env'] in (CoinGame, AsymCoinGame):
        trainable_class = LOLAPGCG
    else:
        trainable_class = LOLAPGMatrice
    print("trainable_class", trainable_class)

    # Train with the Tune Class API (not RLLib Class)
    tune_analysis = tune.run(trainable_class, name=tune_hp["exp_name"], config=tune_config,
                             checkpoint_at_end=True,
                             stop=stop, metric=tune_config["metric"], mode="max")
    tune_analysis_per_exp = {"": tune_analysis}
    return tune_analysis_per_exp


def evaluate(tune_hp, debug, tune_analysis_per_exp):
    (rllib_hp, rllib_config_eval, policies_to_load, trainable_class, stop, env_config) = \
        generate_eval_config(tune_hp, debug)

    evaluate_same_and_cross_perf(rllib_hp, rllib_config_eval, policies_to_load,
                                 trainable_class, stop, env_config, tune_analysis_per_exp)


def evaluate_same_and_cross_perf(rllib_hp, rllib_config_eval, policies_to_load,
                                 trainable_class, stop, env_config, tune_analysis_per_exp):
    evaluator = same_and_cross_perf.SameAndCrossPlayEvaluator(exp_name=rllib_hp["exp_name"])
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=rllib_config_eval, stop_config=stop,
        policies_to_load_from_checkpoint=policies_to_load,
        tune_analysis_per_exp=tune_analysis_per_exp,
        TuneTrainerClass=trainable_class,
        n_cross_play_per_checkpoint=min(5, rllib_hp["train_n_replicates"] - 1),
        to_load_path=rllib_hp["load_plot_data"])

    plot_config = PlotConfig(xlim=rllib_hp["x_limits"], ylim=rllib_hp["y_limits"],
                             markersize=5, alpha=1.0, jitter=rllib_hp["jitter"],
                             xlabel="player 1 payoffs", ylabel="player 2 payoffs",
                             plot_max_n_points=rllib_hp["train_n_replicates"],
                             title="cross and same-play performances: " + rllib_hp['env'].NAME,
                             x_scale_multiplier=rllib_hp["scale_multipliers"][0],
                             y_scale_multiplier=rllib_hp["scale_multipliers"][1])
    evaluator.plot_results(analysis_metrics_per_mode, plot_config=plot_config,
                           x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
                           y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}")


def generate_eval_config(tune_hp, debug):
    rllib_hp = copy.deepcopy(tune_hp)
    rllib_hp["seed"] = 2020
    rllib_hp["num_episodes"] = 1 if debug else 100
    tune_config, stop, env_config = get_tune_config(rllib_hp, stop_on_epi_number=True)
    env_config["batch_size"] = 1
    if tune_config['env'] in (CoinGame, AsymCoinGame):
        tune_config['TuneTrainerClass'] = LOLAPGCG
    else:
        tune_config['TuneTrainerClass'] = LOLAPGMatrice

    rllib_config_eval = {
        "env": rllib_hp["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env"].ACTION_SPACE,
                    {"tune_config": tune_config}),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env"].ACTION_SPACE,
                    {"tune_config": tune_config}),
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

    return rllib_hp, rllib_config_eval, policies_to_load, trainable_class, stop, env_config


def main(debug):
    train_n_replicates = 2 if debug else 40
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("LOLA_PG")

    use_best_exploiter = False
    # use_best_exploiter = True

    high_coop_speed_hp = True if use_best_exploiter else False
    # high_coop_speed_hp = True

    tune_hparams = {
        "debug": debug,
        "exp_name": exp_name,
        "train_n_replicates": train_n_replicates,

        # Print metrics
        "load_plot_data": None,
        # Example: "load_plot_data": ".../SameAndCrossPlay_save.p",

        # Dynamically set
        "num_episodes": 3 if debug else 4000 if high_coop_speed_hp else 2000,
        # "num_episodes": tune.grid_search([2000, 4000, 6000]),
        "trace_length": 4 if debug else 20,
        "lr": None,
        "gamma": 0.5,
        "batch_size": 8 if debug else 512,

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
        "bs_mul": 1 / 10 * 3 if use_best_exploiter else 1 / 10,
        "simple_net": True,
        "hidden": 32,
        "reg": 0,
        "set_zero": 0,

        "exact": False,

        "warmup": 1,

        "seed": tune.grid_search(seeds),

        "changed_config": False,
        "ac_lr": 1.0,
        "summary_len": 1,
        "use_MAE": False,

        # "use_toolbox_env": True,

        "clip_loss_norm": False,
        "clip_lola_update_norm": False,
        "clip_lola_correction_norm": 3.0,
        "clip_lola_actor_norm": 10.0,

        "entropy_coeff": 0.001,

        "weigth_decay": 0.03,

        "lola_correction_multiplier": 1,
        # "lola_correction_multiplier": tune.grid_search([1, 0.75, 0.5, 0.25]),

        "lr_decay": True,

        "correction_reward_baseline_per_step": False,

        "use_critic": False,
    }

    # Add exploiter hyperparameters
    tune_hparams.update({
        "playing_against_exploiter": False,
        # "playing_against_exploiter": True,
        "start_using_exploiter_at_update_n": 1 if debug else 3000 if high_coop_speed_hp else 1500,
        # "use_exploiter_on_fraction_of_batch": 0.5 if debug else 1.0,
        "use_exploiter_on_fraction_of_batch": 0.5 if debug else 0.1,

        # DQN exploiter
        "use_DQN_exploiter": False,
        # "use_DQN_exploiter": True,
        "train_exploiter_n_times_per_epi": 3,
        "exploiter_base_lr": 0.1,
        "exploiter_decay_lr_in_n_epi": 3000 if high_coop_speed_hp else 1500,
        "exploiter_stop_training_after_n_epi": 3000 if high_coop_speed_hp else 1500,
        "exploiter_rolling_avg": 0.9,
        "always_train_PG": True,
        # (is not None) DQN exploiter use thresholds on opp cooperation to switch between policies
        # otherwise the DQN exploiter will use the best policy (from simulated reward)
        # "exploiter_thresholds": None,
        "exploiter_thresholds": [0.6, 0.7] if debug else [0.80, 0.95],

        # PG exploiter
        # "use_PG_exploiter": False,
        "use_PG_exploiter": True if use_best_exploiter else False,
        "every_n_updates_copy_weights": 1 if debug else 100,
        "adding_scaled_weights": False,
        # "adding_scaled_weights": 0.33,

        # Destabilizer exploiter
        "use_destabilizer": True,
        # "use_destabilizer": False,
    })

    if tune_hparams["load_plot_data"] is None:
        ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)
        tune_analysis_per_exp = train(tune_hparams)
    else:
        tune_analysis_per_exp = None

    evaluate(tune_hparams, debug, tune_analysis_per_exp)
    ray.shutdown()

if __name__ == "__main__":
    debug_mode = True
    # debug_mode = False
    main(debug_mode)
