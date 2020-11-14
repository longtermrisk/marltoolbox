##########
# Additional dependency needed:
# A fork of LOLA https://github.com/Manuscrit/lola which corrects some errors in LOLA and add the logging through Tune
# git clone https://github.com/Manuscrit/lola
# pip install -e .
##########

import ray
from lola.envs import *
from ray import tune


def main(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
         lr, lr_correction, batch_size, bs_mul, simple_net, hidden,
         num_units, reg, gamma, lola, opp_model, mem_efficient, seed):
    # Sanity
    assert exp_name in {"CoinGame", "IPD", "IMP", "AsymCoinGame"}

    # Resolve default parameters
    if exact:
        num_episodes = 50 if num_episodes is None else num_episodes
        trace_length = 200 if trace_length is None else trace_length
        lr = 1. if lr is None else lr
    elif exp_name in {"IPD", "IMP"}:
        num_episodes = 600000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        batch_size = 4000 if batch_size is None else batch_size
        lr = 1. if lr is None else lr
    elif exp_name == "CoinGame" or exp_name == "AsymCoinGame":
        num_episodes = 100000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        batch_size = 4000 if batch_size is None else batch_size
        lr = 0.005 if lr is None else lr

    # Import the right training function
    if exact:
        assert exp_name != "CoinGame", "Can't run CoinGame with --exact."
        assert exp_name != "AsymCoinGame", "Can't run AsymCoinGame with --exact."

        def run(env):
            from lola.train_exact import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  simple_net=simple_net,
                  corrections=lola,
                  pseudo=pseudo,
                  num_hidden=hidden,
                  reg=reg,
                  lr=lr,
                  lr_correction=lr_correction,
                  gamma=gamma)
    elif exp_name in {"IPD", "IMP"}:
        def run(env):
            from lola.train_pg import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  gamma=gamma,
                  set_zero=0,
                  lr=lr,
                  corrections=lola,
                  simple_net=simple_net,
                  hidden=hidden,
                  mem_efficient=mem_efficient)
    elif exp_name == "CoinGame":
        def run(env):
            from lola.train_cg import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  bs_mul=bs_mul,
                  gamma=gamma,
                  grid_size=grid_size,
                  lr=lr,
                  corrections=lola,
                  opp_model=opp_model,
                  hidden=hidden,
                  mem_efficient=mem_efficient)
    elif exp_name == "AsymCoinGame":
        def run(env):
            from lola.train_cg import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  bs_mul=bs_mul,
                  gamma=gamma,
                  grid_size=grid_size,
                  lr=lr,
                  corrections=lola,
                  opp_model=opp_model,
                  hidden=hidden,
                  mem_efficient=mem_efficient,
                  asymmetry=True)

    # Instantiate the environment
    if exp_name == "IPD":
        env = IPD(trace_length)
        gamma = 0.96 if gamma is None else gamma
    elif exp_name == "IMP":
        env = IMP(trace_length)
        gamma = 0.9 if gamma is None else gamma
    elif exp_name == "CoinGame":
        env = CG(trace_length, batch_size, grid_size)
        gamma = 0.96 if gamma is None else gamma
    elif exp_name == "AsymCoinGame":
        env = AsymCG(trace_length, batch_size, grid_size)
        gamma = 0.96 if gamma is None else gamma

    if exp_name == "CoinGame" or exp_name == "AsymCoinGame":
        env.seed(seed)
    run(env)


def lola_training(config):
    main(**config)


if __name__ == "__main__":

    # exp_name = "IPD"
    exp_name = "CoinGame"

    ray.init(num_cpus=2, num_gpus=0)

    full_config = {
            "exp_name": exp_name,
            "num_episodes": None,
            "trace_length": None,
            "pseudo": False,
            "grid_size": 3,
            "lola": True,
            "opp_model": False,
            "mem_efficient": True,
            "lr": None,
            "lr_correction": 1,
            "bs_mul": 1,
            "simple_net": True,
            "hidden": 32,
            "num_units": 64,
            "reg": 0,
            "gamma": None,

            # "exact": True,
            "exact": False,

            # !!! To use the default batch size with coin game, you need 35Go of memory per seed run in parallel !!!
            # "batch_size": None, # To use the defaults values from the official repository.
            "batch_size": 10,

            "seed": tune.grid_search([1, 2, 3, 4, 5]),
        }

    analysis = tune.run(lola_training, name=f"LOLA_{exp_name}", config=full_config)

    # If needed, get a dataframe for analyzing trial results.
    df = analysis.results_df
