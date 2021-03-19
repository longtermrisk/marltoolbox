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
from marltoolbox.envs import \
    vectorized_coin_game, vectorized_mixed_motive_coin_game, \
    matrix_sequential_social_dilemma
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import policy, log, self_and_cross_perf
from marltoolbox.utils.plot import PlotConfig


def main(debug, env=None):
    train_n_replicates = 2 if debug else 40
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("LOLA_PG")

    # The InfluenceEvader(like)
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
        # Example: "load_plot_data": ".../SelfAndCrossPlay_save.p",

        # Dynamically set
        "num_episodes": 3 if debug else 4000 if high_coop_speed_hp else 2000,
        "trace_length": 4 if debug else 20,
        "lr": None,
        "gamma": 0.5,
        "batch_size": 8 if debug else 512,

        # "env_name": "IteratedPrisonersDilemma" if env is None else env,
        # "env_name": "IteratedAsymBoS" if env is None else env,
        "env_name": "VectorizedCoinGame" if env is None else env,
        # "env_name": "AsymVectorizedCoinGame" if env is None else env,
        # "env_name": "VectorizedMixedMotiveCoinGame" if env is None else env,

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

        "plot_keys": ["reward", "total_reward", "entrop", ],
        "plot_assemblage_tags": [("total_reward",), ("entrop",), ],
    }

    # Add exploiter hyperparameters
    tune_hparams.update({
        "playing_against_exploiter": False,
        # "playing_against_exploiter": True,
        "start_using_exploiter_at_update_n":
            1 if debug else 3000 if high_coop_speed_hp else 1500,
        # "use_exploiter_on_fraction_of_batch": 0.5 if debug else 1.0,
        "use_exploiter_on_fraction_of_batch": 0.5 if debug else 0.1,

        # DQN exploiter
        # TODO remove this (the DQN exploiter versions)
        "use_DQN_exploiter": False,
        # "use_DQN_exploiter": True,
        "train_exploiter_n_times_per_epi": 3,
        "exploiter_base_lr": 0.1,
        "exploiter_decay_lr_in_n_epi":
            3000 if high_coop_speed_hp else 1500,
        "exploiter_stop_training_after_n_epi":
            3000 if high_coop_speed_hp else 1500,
        "exploiter_rolling_avg": 0.9,
        "always_train_PG": True,
        # (if not None) DQN exploiter use thresholds on opp cooperation to
        # switch between policies
        # otherwise the DQN exploiter will use the best policy
        # (from simulated reward)
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


def train(tune_hp):
    tune_config, stop, env_config = get_tune_config(tune_hp)

    if "CoinGame" in tune_config['env_name']:
        trainable_class = LOLAPGCG
    else:
        trainable_class = LOLAPGMatrice

    # Train with the Tune Class API (not RLLib Class)
    tune_analysis = tune.run(trainable_class, name=tune_hp["exp_name"],
                             config=tune_config,
                             checkpoint_at_end=True,
                             stop=stop, metric=tune_config["metric"],
                             mode="max")
    tune_analysis_per_exp = {"": tune_analysis}

    # if not tune_hp["debug"]:
    aggregate_and_plot_tensorboard_data.add_summary_plots(
        main_path=os.path.join("~/ray_results/", tune_config["exp_name"]),
        plot_keys=tune_config["plot_keys"],
        plot_assemble_tags_in_one_plot=tune_config["plot_assemblage_tags"],
    )

    return tune_analysis_per_exp


def get_tune_config(tune_hp: dict, stop_on_epi_number: bool = False):
    tune_config = copy.deepcopy(tune_hp)

    assert not tune_config['exact']

    # Resolve default parameters
    if "CoinGame" in tune_config['env_name']:
        if tune_config['env_name'] == "VectorizedCoinGame":
            tune_config['env_class'] = \
                vectorized_coin_game.VectorizedCoinGame
        elif tune_config['env_name'] == "AsymVectorizedCoinGame":
            tune_config['env_class'] = \
                vectorized_coin_game.AsymVectorizedCoinGame
        elif tune_config['env_name'] == "VectorizedMixedMotiveCoinGame":
            tune_config['env_class'] = \
                vectorized_mixed_motive_coin_game.VectorizedMixedMotiveCoinGame
        else:
            raise ValueError()

        tune_config['num_episodes'] = 100000 \
            if tune_config['num_episodes'] is None \
            else tune_config['num_episodes']
        tune_config['trace_length'] = 150 \
            if tune_config['trace_length'] is None \
            else tune_config['trace_length']
        tune_config['batch_size'] = 4000 \
            if tune_config['batch_size'] is None \
            else tune_config['batch_size']
        tune_config['lr'] = 0.005 \
            if tune_config['lr'] is None \
            else tune_config['lr']
        tune_config['gamma'] = 0.96 \
            if tune_config['gamma'] is None \
            else tune_config['gamma']
        tune_hp["x_limits"] = (-1.0, 1.0)
        tune_hp["y_limits"] = (-1.0, 1.0)
        if tune_config['env_class'] == \
                vectorized_coin_game.AsymVectorizedCoinGame:
            tune_hp["x_limits"] = (-1.0, 3.0)
        tune_hp["jitter"] = 0.02
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "grid_size": tune_config["grid_size"],
            "get_additional_info": True,
            "both_players_can_pick_the_same_coin":
                tune_config['env_name'] == "VectorizedMixedMotiveCoinGame",
            "force_vectorize": False,
            "same_obs_for_each_player": True,
        }
        tune_config['metric'] = "player_blue_pick_speed"
        tune_config["plot_keys"] += \
            ["speed", "own_color", ] + vectorized_coin_game.PLOT_KEYS
        tune_config["plot_assemblage_tags"] += \
            [("own",), ("own_color",), ("speed",), ("pick_speed",), ] + \
            vectorized_coin_game.PLOT_ASSEMBLAGE_TAGS
    else:
        if tune_config['env_name'] == "IteratedPrisonersDilemma":
            tune_config['env_class'] = \
                matrix_sequential_social_dilemma.IteratedPrisonersDilemma
        elif tune_config['env_name'] == "IteratedAsymBoS":
            tune_config['env_class'] = \
                matrix_sequential_social_dilemma.IteratedAsymBoS
        else:
            raise ValueError()

        tune_config['num_episodes'] = 600000 \
            if tune_config['num_episodes'] is None \
            else tune_config['num_episodes']
        tune_config['trace_length'] = 150 \
            if tune_config['trace_length'] is None \
            else tune_config['trace_length']
        tune_config['batch_size'] = 4000 \
            if tune_config['batch_size'] is None \
            else tune_config['batch_size']
        tune_config['lr'] = 1.0 \
            if tune_config['lr'] is None \
            else tune_config['lr']
        tune_config['gamma'] = 0.96 \
            if tune_config['gamma'] is None \
            else tune_config['gamma']
        tune_hp["x_limits"] = (-3.0, 3.0)
        tune_hp["y_limits"] = (-3.0, 3.0)
        tune_hp["jitter"] = 0.05
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "batch_size": tune_config["batch_size"],
            "max_steps": tune_config["trace_length"],
            "get_additional_info": True,
        }
        tune_config['metric'] = "player_row_CC_freq"

    tune_hp["scale_multipliers"] = (
        1 / tune_config['trace_length'], 1 / tune_config['trace_length'])
    tune_config["env_config"] = env_config

    if stop_on_epi_number:
        stop = {"episodes_total": tune_config['num_episodes']}
    else:
        stop = {"finished": True}

    return tune_config, stop, env_config


def evaluate(tune_hp, debug, tune_analysis_per_exp):
    (rllib_hp, rllib_config_eval, policies_to_load, trainable_class, stop,
     env_config) = \
        generate_eval_config(tune_hp, debug)

    evaluate_self_and_cross_perf(rllib_hp, rllib_config_eval, policies_to_load,
                                 trainable_class, stop, env_config,
                                 tune_analysis_per_exp)


def generate_eval_config(tune_hp, debug):
    rllib_hp = copy.deepcopy(tune_hp)
    rllib_hp["seed"] = 2020
    rllib_hp["num_episodes"] = 1 if debug else 100
    tune_config, stop, env_config = get_tune_config(rllib_hp,
                                                    stop_on_epi_number=True)
    rllib_hp["env_class"] = tune_config["env_class"]

    if "CoinGame" in tune_config['env_name']:
        env_config["batch_size"] = 1
        tune_config['TuneTrainerClass'] = LOLAPGCG
    else:
        tune_config['TuneTrainerClass'] = LOLAPGMatrice

    rllib_config_eval = {
        "env": rllib_hp["env_class"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    # The default policy is DQN defined in DQNTrainer
                    # but we overwrite it to use the LE policy
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env_class"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env_class"].ACTION_SPACE,
                    {"tune_config": tune_config}),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env_class"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env_class"].ACTION_SPACE,
                    {"tune_config": tune_config}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": rllib_hp["seed"],
        "min_iter_time_s": 3.0,
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])

    if "CoinGame" in rllib_hp['env_name']:
        trainable_class = LOLAPGCG
        rllib_config_eval["model"] = {
            "dim": env_config["grid_size"],
            # [Channel, [Kernel, Kernel], Stride]]
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
        }
    else:
        trainable_class = LOLAPGMatrice
        # rllib_config_eval["model"] = {
        #     # Number of hidden layers for fully connected net
        #     "fcnet_hiddens": tune_hp["hidden"],
        #     # Nonlinearity for fully connected net (tanh, relu)
        #     "fcnet_activation": "relu",
        # }

    return rllib_hp, rllib_config_eval, policies_to_load, \
           trainable_class, stop, env_config


def evaluate_self_and_cross_perf(
        rllib_hp, rllib_config_eval, policies_to_load, trainable_class,
        stop, env_config, tune_analysis_per_exp):
    evaluator = self_and_cross_perf.SelfAndCrossPlayEvaluator(
        exp_name=rllib_hp["exp_name"])
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=rllib_config_eval, stop_config=stop,
        policies_to_load_from_checkpoint=policies_to_load,
        tune_analysis_per_exp=tune_analysis_per_exp,
        TuneTrainerClass=trainable_class,
        n_cross_play_per_checkpoint=min(5, rllib_hp["train_n_replicates"] - 1),
        to_load_path=rllib_hp["load_plot_data"])

    if issubclass(
            rllib_hp['env_class'],
            matrix_sequential_social_dilemma.MatrixSequentialSocialDilemma):
        background_area_coord = rllib_hp['env_class'].PAYOUT_MATRIX
    else:
        background_area_coord = None

    plot_config = PlotConfig(
        xlim=rllib_hp["x_limits"],
        ylim=rllib_hp["y_limits"],
        markersize=5, alpha=1.0,
        jitter=rllib_hp["jitter"],
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        plot_max_n_points=rllib_hp["train_n_replicates"],
        x_scale_multiplier=rllib_hp["scale_multipliers"][0],
        y_scale_multiplier=rllib_hp["scale_multipliers"][1],
        background_area_coord=background_area_coord)
    evaluator.plot_results(
        analysis_metrics_per_mode,
        plot_config=plot_config,
        x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}")


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
