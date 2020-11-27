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

import ray
from ray import tune
from marltoolbox.utils import experimenting
from ray.rllib.agents.pg import DEFAULT_CONFIG
from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame

from lola.train_cg_tune_class_API import LOLAPGTrainableClass

def dynamically_change_config(full_config: dict) -> dict:
    # Sanity
    assert full_config['exp_name'] in {"CoinGame","AsymCoinGame"}
    assert not full_config['exact']

    # Resolve default parameters
    full_config['num_episodes'] = 100000 if full_config['num_episodes'] is None else full_config['num_episodes']
    full_config['trace_length'] = 150 if full_config['trace_length'] is None else full_config['trace_length']
    full_config['batch_size'] = 4000 if full_config['batch_size'] is None else full_config['batch_size']
    full_config['lr'] = 0.005 if full_config['lr'] is None else full_config['lr']
    full_config['gamma'] = 0.96 if full_config['gamma'] is None else full_config['gamma']

    return full_config


if __name__ == "__main__":
    debug = True

    full_config = {
        # Dynamically set
        "num_episodes": None,
        # "trace_length": None,
        "trace_length": 20 if debug else None,
        "lr": None,
        # "lr": 0.005 / 10,  # None,
        "gamma": 0.5 if debug else None,
        # "gamma": 0.9,
        # !!! To use the default batch size with coin game, you need 35Go of memory per seed run in parallel !!!
        # "batch_size": None, # To use the defaults values from the official repository.
        # "batch_size": 100,
        "batch_size": 64 if debug else None,

        # "exp_name": "CoinGame",
        "exp_name": "AsymCoinGame",

        "pseudo": False,
        "grid_size": 3,
        "lola_update": True,
        "opp_model": False,
        "mem_efficient": True,
        "lr_correction": 1,
        "bs_mul": 1,
        "simple_net": True,
        "hidden": 32,
        "reg": 0,
        "set_zero": 0,

        # "exact": True,
        "exact": False,

        "warmup": 1,  #False,

        "run_n_seed_in_parallel": 2,
        "seed": tune.grid_search([1,2]),

        "changed_config": False,
        "ac_lr": 0.005 if debug else 1.0,
        # "ac_lr": 0.05,
        "summary_len": 1,
        "use_MAE": False,
        # "use_MAE": True,

        "perform_lola_update": True,
        # "perform_lola_update": False,

        # "use_toolbox_env": False,
        "use_toolbox_env": True,

        "clip_lola_update_norm":False,
        "clip_loss_norm":False,
        # "clip_lola_update_norm": 5.0,
        # "clip_loss_norm": 10.0,
    }

    full_config = dynamically_change_config(full_config)

    full_config['TuneTrainerClass'] = LOLAPGTrainableClass
    full_config['seed'] = 1
    env_config = {
        "batch_size": full_config["batch_size"],
        "max_steps": full_config["trace_length"],
        "grid_size": full_config["grid_size"],
        "get_additional_info": True,
    }

    # p = experimenting.FixedPolicyFromTuneTrainer(
    #     CoinGame(env_config).OBSERVATION_SPACE,
    #     CoinGame.ACTION_SPACE,
    #     copy.deepcopy(full_config))
    # p.tune_trainer.cleanup()
    # del p
    # full_config['seed'] = 3
    # p = experimenting.FixedPolicyFromTuneTrainer(
    #     CoinGame(env_config).OBSERVATION_SPACE,
    #     CoinGame.ACTION_SPACE,
    #     copy.deepcopy(full_config))
    # full_config['seed'] = 2
    # p = experimenting.FixedPolicyFromTuneTrainer(
    #     CoinGame(env_config).OBSERVATION_SPACE,
    #     CoinGame.ACTION_SPACE,
    #     copy.deepcopy(full_config))

    stop = {"episodes_total": 10 if debug else 10000}

    ray.init(num_cpus=full_config["run_n_seed_in_parallel"], num_gpus=0)
    name = f"LOLA_{'exact' if full_config['exact'] else 'PG'}_{full_config['exp_name']}"
    results = tune.run(LOLAPGTrainableClass, name=name, config=full_config,
                       checkpoint_at_end=True, #checkpoint_freq=10,
                       stop=stop, metric="player_blue_pick_own", mode="max")


    # ray.shutdown()
    # ray.init(num_cpus=full_config["run_n_seed_in_parallel"], num_gpus=0)


    # Evaluate
    full_config.pop("seed")
    full_config["batch_size"] = 1
    env_config = {
        "batch_size": full_config["batch_size"],
        "max_steps": full_config["trace_length"],
        "grid_size": full_config["grid_size"],
        "get_additional_info": True,
    }

    eval_config_update = copy.deepcopy(DEFAULT_CONFIG)
    eval_config_update.update({
        "env": CoinGame,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "player_red": (
                    # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
                    experimenting.FixedPolicyFromTuneTrainer,
                    CoinGame(env_config).OBSERVATION_SPACE,
                    CoinGame.ACTION_SPACE,
                    copy.deepcopy(full_config)),
                "player_blue": (
                    experimenting.FixedPolicyFromTuneTrainer,
                    CoinGame(env_config).OBSERVATION_SPACE,
                    CoinGame.ACTION_SPACE,
                    copy.deepcopy(full_config)),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": [],
        },
        "seed": 3,
    })

    stop = {"episodes_total": 100}

    experiment = experimenting.SameAndCrossPlay(TuneTrainerClass=LOLAPGTrainableClass,
                                                extract_checkpoints_from_results=results,
                                                evaluation_config=eval_config_update,
                                                stop_config=stop,
                                                exp_name="LOLA_eval",
                                                n_same_play_per_checkpoint=1,
                                                n_cross_play_per_checkpoint=0)

    ray.shutdown()
