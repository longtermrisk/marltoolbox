##########
# Additional dependency needed:
# A fork of LOLA https://github.com/Manuscrit/lola which corrects some errors in LOLA and add the logging through Tune
# git clone https://github.com/Manuscrit/lola
# pip install -e .
##########

from datetime import datetime

import ray
from ray import tune

import tensorflow as tf

import lola_dice.envs
from lola_dice.policy import SimplePolicy, MLPPolicy, ConvPolicy
from lola_dice.rpg import train


def make_simple_policy(ob_size, num_actions, prev=None, root=None):
    return SimplePolicy(ob_size, num_actions, prev=prev)


def make_mlp_policy(ob_size, num_actions, prev=None):
    return MLPPolicy(ob_size, num_actions, hidden_sizes=[64], prev=prev)


def make_conv_policy(ob_size, num_actions, prev=None):
    return ConvPolicy(ob_size, num_actions, hidden_sizes=[16, 32], prev=prev)


def make_adam_optimizer(*, lr):
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                  name='Adam')


def make_sgd_optimizer(*, lr):
    return tf.train.GradientDescentOptimizer(learning_rate=lr)


def main(use_dice, use_opp_modeling, epochs, batch_size, runs, exp_name, trace_length, gamma, grid_size,
         lr_inner, lr_outer, lr_value, lr_om, inner_asymm, n_agents, n_inner_steps, value_batch_size,
         value_epochs, om_batch_size, om_epochs, use_baseline, save_dir):
    # if exp_name in {"IPD", "IMP"}:
    #     make_optim = make_sgd_optimizer
    #
    # elif exp_name == "CoinGame":
    #     make_optim = make_adam_optimizer
    #
    # if exp_name in {"IPD", "IMP"}:
    #     make_optim = make_sgd_optimizer
    #
    # elif exp_name == "CoinGame":
    #     make_optim = make_adam_optimizer
    #
    # # Instantiate the environment
    # if exp_name == "IPD":
    #     policy_maker = make_simple_policy
    # elif exp_name == "IMP":
    #     policy_maker = make_simple_policy
    # elif exp_name == "CoinGame":
    #     policy_maker = make_conv_policy

    # Instantiate the environment
    if exp_name == "IPD":
        env = lola_dice.envs.IPD(max_steps=trace_length, batch_size=batch_size)
    elif exp_name == "IMP":
        env = lola_dice.envs.IMP(trace_length)
    elif exp_name == "CoinGame":
        env = lola_dice.envs.CG(trace_length, batch_size, grid_size)
        timestamp = datetime.now().timestamp()
        env.seed(int(timestamp))
    # elif exp_name == "AsymCoinGame":
    #     env = AsymCG(trace_length, batch_size, grid_size)
    #     gamma = 0.96 if gamma is None else gamma

    else:
        raise ValueError(f"exp_name: {exp_name}")

    train(env, policy_maker,
          make_optim,
          epochs=epochs,
          gamma=gamma,
          lr_inner=lr_inner,
          lr_outer=lr_outer,
          lr_value=lr_value,
          lr_om=lr_om,
          inner_asymm=inner_asymm,
          n_agents=n_agents,
          n_inner_steps=n_inner_steps,
          value_batch_size=value_batch_size,
          value_epochs=value_epochs,
          om_batch_size=om_batch_size,
          om_epochs=om_epochs,
          use_baseline=use_baseline,
          use_dice=use_dice,
          use_opp_modeling=use_opp_modeling)


def lola_training(config):
    main(**config)


def dynamically_change_config(full_config: dict) -> dict:
    if full_config["exp_name"] == "IPD":
        full_config["gamma"] = 0.96 if full_config["gamma"] is None else full_config["gamma"]
        full_config["policy_maker"] = make_simple_policy
        full_config["base_lr"] = 1.0
        full_config["num_episodes"] = 600000 if full_config["num_episodes"] is None else full_config["num_episodes"]
        full_config["trace_length"] = 150 if full_config["trace_length"] is None else full_config["trace_length"]
        full_config["batch_size"] = 4000 if full_config["batch_size"] is None else full_config["batch_size"]
        full_config["make_optim"] = make_sgd_optimizer
        full_config["save_dir"] = "dice_results_ipd"
    elif full_config["exp_name"] == "IMP":
        full_config["gamma"] = 0.9 if full_config["gamma"] is None else full_config["gamma"]
        full_config["policy_maker"] = make_simple_policy
        full_config["base_lr"] = 1.0
        full_config["num_episodes"] = 600000 if full_config["num_episodes"] is None else full_config["num_episodes"]
        full_config["trace_length"] = 150 if full_config["trace_length"] is None else full_config["trace_length"]
        full_config["batch_size"] = 4000 if full_config["batch_size"] is None else full_config["batch_size"]
        full_config["make_optim"] = make_sgd_optimizer
        full_config["save_dir"] = "dice_results_imp"
    elif full_config["exp_name"] == "CoinGame":
        full_config["num_episodes"] = 100000 if full_config["num_episodes"] is None else full_config["num_episodes"]
        full_config["trace_length"] = 150 if full_config["trace_length"] is None else full_config["trace_length"]
        full_config["batch_size"] = 4000 if full_config["batch_size"] is None else full_config["batch_size"]
        full_config["epochs"] *= 10
        full_config["make_optim"] = make_adam_optimizer
        full_config["save_dir"] = "dice_results_coin_game"
        full_config["gamma"] = 0.96 if full_config["gamma"] is None else full_config["gamma"]
        full_config["policy_maker"] = make_conv_policy
        full_config["base_lr"] = 0.005

    full_config["lr_inner"] = full_config["lr_inner"] * full_config["base_lr"]
    full_config["lr_outer"] = full_config["lr_outer"] * full_config["base_lr"]
    full_config["lr_value"] = full_config["lr_value"] * full_config["base_lr"]
    full_config["lr_om"] = full_config["lr_om"] * full_config["base_lr"]

    return full_config


if __name__ == "__main__":
    run_n_seed_in_parallel = 2
    ray.init(num_cpus=run_n_seed_in_parallel, num_gpus=0)

    full_config = {
        # "exp_name": "IPD",
        # "exp_name": "IMP",
        # "exp_name": "CoinGame",
        "exp_name": "AsymCoinGame",

        "gamma": None,
        "num_episodes": None,
        "trace_length": None,

        "batch-size": 64,
        "epochs": 200,
        "lr_inner": .1,
        "lr_outer": .2,
        "lr_value": .1,
        "lr_om": .1,
        "inner_asymm": True,
        "n_agents": 2,
        "n_inner_steps": 2,
        "value_batch_size": 16,
        "value_epochs": 0,
        "om_batch_size": 16,
        "om_epochs": 0,
        "grid_size":3,
        "use_baseline": False,
        "use_dice": True,
        "use_opp_modeling": False,

        "seed": tune.grid_search([1, 2, 3, 4, 5]),
    }

    full_config = dynamically_change_config(full_config)

    analysis = tune.run(lola_training, name=f"LOLA_{full_config['exp_name']}", config=full_config)

    # If needed, get a dataframe for analyzing trial results.
    df = analysis.results_df
