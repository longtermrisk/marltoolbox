##########
# Additional dependencies are needed:
# 1) Python 3.6
# conda install python=3.6
# 2) A fork of LOLA https://github.com/Manuscrit/lola which adds the logging through Tune
# git clone https://github.com/Manuscrit/lola
# git checkout d9c6724ea0d6bca42c8cf9688b1ff8d6fefd7267
# pip install -e .
##########

# TODO only used envs from lola_dice
import lola.envs
import lola_dice.envs
import lola.train_cg
import lola.train_exact
import lola.train_pg
import ray
from ray import tune


def main(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
         lr, lr_correction, batch_size, bs_mul, simple_net, hidden, reg,
         gamma, lola_update, opp_model, mem_efficient, seed, set_zero, **kwargs):
    # Instantiate the environment
    if exp_name == "IPD":
        env = lola.envs.IPD(trace_length)
    elif exp_name == "IMP":
        env = lola.envs.IMP(trace_length)
    elif exp_name == "CoinGame":
        env = lola_dice.envs.CG(trace_length, batch_size, grid_size)
        env.seed(seed)
    elif exp_name == "AsymCoinGame":
        env = lola_dice.envs.AsymCG(trace_length, batch_size, grid_size)
        env.seed(seed)
    else:
        raise ValueError(f"exp_name: {exp_name}")

    # Import the right training function
    if exact:
        lola.train_exact.train(env,
                         num_episodes=num_episodes,
                         trace_length=trace_length,
                         simple_net=simple_net,
                         corrections=lola_update,
                         pseudo=pseudo,
                         num_hidden=hidden,
                         reg=reg,
                         lr=lr,
                         lr_correction=lr_correction,
                         gamma=gamma)
    elif exp_name in ("IPD", "IMP"):
        lola.train_pg.train(env,
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
        lola.train_cg.train(env,
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
                      asymmetry=exp_name == "AsymCoinGame")
    else:
        raise ValueError(f"exp_name: {exp_name}")

def lola_training(config):
    main(**config)


def dynamically_change_config(full_config: dict) -> dict:
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


if __name__ == "__main__":

    full_config = {
        # Dynamically set
        "num_episodes": None,
        "trace_length": None,
        "lr": None,
        "gamma": None,
        # !!! To use the default batch size with coin game, you need 35Go of memory per seed run in parallel !!!
        # "batch_size": None, # To use the defaults values from the official repository.
        "batch_size": 10,

        # "exp_name": "IPD",
        # "exp_name": "IMP",
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

        "run_n_seed_in_parallel": 1,
        "seed": tune.grid_search([1]),
    }

    full_config = dynamically_change_config(full_config)

    ray.init(num_cpus=full_config["run_n_seed_in_parallel"], num_gpus=0)
    name = f"LOLA_{'exact' if full_config['exact'] else 'PG'}_{full_config['exp_name']}"
    analysis = tune.run(lola_training, name=name, config=full_config)

    # If needed, get a dataframe for analyzing trial results.
    df = analysis.results_df
