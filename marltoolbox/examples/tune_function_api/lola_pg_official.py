##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the tune_class_api/lola_pg_official.py file
##########

import os
import ray
from ray import tune

import marltoolbox.algos.lola_dice.envs as lola_dice_envs
from marltoolbox.algos.lola import train_cg, train_pg
from marltoolbox.envs.vectorized_coin_game import CoinGame, AsymCoinGame
from marltoolbox.utils import log


def trainer_fn(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
         lr, lr_correction, batch_size, bs_mul, simple_net, hidden, reg,
         gamma, lola_update, opp_model, mem_efficient, seed, set_zero,
         warmup, changed_config, ac_lr, summary_len, use_MAE,
         use_toolbox_env, clip_lola_update_norm, clip_loss_norm, entropy_coeff,
         weigth_decay, **kwargs):
    # Instantiate the environment
    if exp_name == "IPD":
        raise NotImplementedError()
    elif exp_name == "IMP":
        raise NotImplementedError()
    elif exp_name == "CoinGame":
        if use_toolbox_env:
            env = CoinGame(config={
                "batch_size": batch_size,
                "max_steps": trace_length,
                "grid_size": grid_size,
                "get_additional_info": True,
                "add_position_in_epi": False,
            })
        else:
            env = lola_dice_envs.CG(trace_length, batch_size, grid_size)
        env.seed(seed)
    elif exp_name == "AsymCoinGame":
        if use_toolbox_env:
            env = AsymCoinGame(config={
                "batch_size": batch_size,
                "max_steps": trace_length,
                "grid_size": grid_size,
                "get_additional_info": True,
                "add_position_in_epi": False,
            })
        else:
            env = lola_dice_envs.AsymCG(trace_length, batch_size, grid_size)
        env.seed(seed)
    else:
        raise ValueError(f"exp_name: {exp_name}")

    # Import the right training function
    if exact:
        raise NotImplementedError()
    elif exp_name in ("IPD", "IMP"):
        train_pg.train(env,
                       num_episodes=num_episodes,
                       trace_length=trace_length,
                       batch_size=batch_size,
                       gamma=gamma,
                       set_zero=set_zero,
                       lr=lr,
                       corrections=lola_update,
                       simple_net=simple_net,
                       hidden=hidden,
                       mem_efficient=mem_efficient)
    elif exp_name in ("CoinGame", "AsymCoinGame"):
        train_cg.train(env,
                       num_episodes=num_episodes,
                       trace_length=trace_length,
                       batch_size=batch_size,
                       bs_mul=bs_mul,
                       gamma=gamma,
                       grid_size=grid_size,
                       lr=lr,
                       corrections=lola_update,
                       opp_model=opp_model,
                       hidden=hidden,
                       mem_efficient=mem_efficient,
                       asymmetry=exp_name == "AsymCoinGame",
                       warmup=warmup,
                       changed_config=changed_config,
                       ac_lr=ac_lr,
                       summary_len=summary_len,
                       use_MAE=use_MAE,
                       use_toolbox_env=use_toolbox_env,
                       clip_lola_update_norm=clip_lola_update_norm,
                       clip_loss_norm=clip_loss_norm,
                       entropy_coeff=entropy_coeff,
                       weigth_decay=weigth_decay,
                       )
    else:
        raise ValueError(f"exp_name: {exp_name}")


def lola_training(config):
    trainer_fn(**config)


def get_tune_config(full_config: dict) -> dict:
    # Sanity
    assert full_config['exp_name'] in {"CoinGame", "IPD", "IMP", "AsymCoinGame"}
    if full_config['exact']:
        assert full_config['exp_name'] != "CoinGame", "Can't run CoinGame with --exact."
        assert full_config['exp_name'] != "AsymCoinGame", "Can't run AsymCoinGame with --exact."

    # Resolve default parameters
    if full_config['exact']:
        full_config['num_episodes'] = 50 if full_config['num_episodes'] is None else full_config['num_episodes']
        full_config['trace_length'] = 200 if full_config['trace_length'] is None else full_config['trace_length']
        full_config['lr'] = 1. if full_config['lr'] is None else full_config['lr']
    elif full_config['exp_name'] in {"IPD", "IMP"}:
        full_config['num_episodes'] = 600000 if full_config['num_episodes'] is None else full_config['num_episodes']
        full_config['trace_length'] = 150 if full_config['trace_length'] is None else full_config['trace_length']
        full_config['batch_size'] = 4000 if full_config['batch_size'] is None else full_config['batch_size']
        full_config['lr'] = 1. if full_config['lr'] is None else full_config['lr']
    elif full_config['exp_name'] == "CoinGame" or full_config['exp_name'] == "AsymCoinGame":
        full_config['num_episodes'] = 100000 if full_config['num_episodes'] is None else full_config['num_episodes']
        full_config['trace_length'] = 150 if full_config['trace_length'] is None else full_config['trace_length']
        full_config['batch_size'] = 4000 if full_config['batch_size'] is None else full_config['batch_size']
        full_config['lr'] = 0.005 if full_config['lr'] is None else full_config['lr']

    if full_config['exp_name'] in ("IPD", "CoinGame", "AsymCoinGame"):
        full_config['gamma'] = 0.96 if full_config['gamma'] is None else full_config['gamma']
    elif full_config['exp_name'] == "IMP":
        full_config['gamma'] = 0.9 if full_config['gamma'] is None else full_config['gamma']

    return full_config


def main(debug):
    exp_name, _ = log.log_in_current_day_dir(f"LOLA_PG")

    tune_hparams = {
        "exp_name": exp_name,

        # Dynamically set
        "num_episodes": 3 if debug else None,
        "trace_length": 6 if debug else None,
        "lr": None,
        "gamma": None,
        "batch_size": 12 if debug else None,

        # "exp_name": "IPD",
        # "exp_name": "IMP",
        "exp_name": "CoinGame",
        # "exp_name": "AsymCoinGame",

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

        "seed": 1,

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
    }
    tune_config = get_tune_config(tune_hparams)

    ray.init(num_cpus=os.cpu_count(), num_gpus=0)
    tune_analysis = tune.run(lola_training, name=tune_hparams["exp_name"], config=tune_config)
    ray.shutdown()

    return tune_analysis

if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
