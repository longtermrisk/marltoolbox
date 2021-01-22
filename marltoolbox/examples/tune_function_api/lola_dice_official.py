##########
# Additional dependencies are needed:
# Follow the LOLA installation described in the tune_class_api/lola_pg_official.py file
##########

import copy

import inspect
import os
import ray
import tensorflow as tf
from datetime import datetime
from ray import tune

import marltoolbox.algos.lola_dice.envs as lola_dice_envs
from marltoolbox.algos.lola_dice.policy import SimplePolicy, MLPPolicy, ConvPolicy
from marltoolbox.algos.lola_dice.rpg import train
from marltoolbox.utils import log


def make_simple_policy(ob_size, num_actions, prev=None, root=None, batch_size=None):
    return SimplePolicy(ob_size, num_actions, prev=prev)


def make_mlp_policy(ob_size, num_actions, prev=None, batch_size=64):
    return MLPPolicy(ob_size, num_actions, hidden_sizes=[64], prev=prev, batch_size=batch_size)


def make_conv_policy(ob_size, num_actions, prev=None, batch_size=64):
    return ConvPolicy(ob_size, num_actions, hidden_sizes=[16, 32], prev=prev, batch_size=batch_size)


def make_adam_optimizer(*, lr):
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                  name='Adam')


def make_sgd_optimizer(*, lr):
    return tf.train.GradientDescentOptimizer(learning_rate=lr)


def trainer_fn(use_dice, use_opp_modeling, epochs, batch_size, env_name, trace_length, gamma, grid_size,
               lr_inner, lr_outer, lr_value, lr_om, inner_asymm, n_agents, n_inner_steps, value_batch_size,
               value_epochs, om_batch_size, om_epochs, use_baseline, policy_maker, make_optim, **kwargs):
    # Instantiate the environment
    if env_name == "IPD":
        env = lola_dice_envs.IPD(max_steps=trace_length, batch_size=batch_size)
    elif env_name == "IMP":
        env = lola_dice_envs.IMP(trace_length)
    elif env_name == "CoinGame":
        env = lola_dice_envs.CG(trace_length, batch_size, grid_size)
        timestamp = datetime.now().timestamp()
        env.seed(int(timestamp))
    elif env_name == "AsymCoinGame":
        env = lola_dice_envs.AsymCG(trace_length, batch_size, grid_size)
        timestamp = datetime.now().timestamp()
        env.seed(int(timestamp))
    else:
        raise ValueError(f"env_name: {env_name}")

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
    trainer_fn(**config)


def get_tune_config(tune_hp: dict) -> dict:
    tune_config = copy.deepcopy(tune_hp)
    assert tune_config['env_name'] in {"CoinGame", "IPD", "IMP", "AsymCoinGame"}

    if tune_config["env_name"] in ("IPD", "IMP"):
        tune_config["policy_maker"] = make_simple_policy
        tune_config["base_lr"] = 1.0
        tune_config["trace_length"] = 150 if tune_config["trace_length"] is None else tune_config["trace_length"]
        tune_config["make_optim"] = make_sgd_optimizer

    if tune_config["env_name"] == "IPD":
        tune_config["gamma"] = 0.96 if tune_config["gamma"] is None else tune_config["gamma"]
        tune_config["save_dir"] = "dice_results_ipd"
    elif tune_config["env_name"] == "IMP":
        tune_config["gamma"] = 0.9 if tune_config["gamma"] is None else tune_config["gamma"]
        tune_config["save_dir"] = "dice_results_imp"
    elif tune_config["env_name"] in ("CoinGame", "AsymCoinGame"):
        tune_config["trace_length"] = 150 if tune_config["trace_length"] is None else tune_config["trace_length"]
        tune_config["epochs"] = int(tune_config["epochs"] * 10)
        tune_config["make_optim"] = make_adam_optimizer
        tune_config["save_dir"] = "dice_results_coin_game"
        tune_config["gamma"] = 0.96 if tune_config["gamma"] is None else tune_config["gamma"]
        tune_config["policy_maker"] = make_conv_policy
        tune_config["base_lr"] = 0.005

    tune_config["make_optim_source"] = inspect.getsource(tune_config["make_optim"])
    tune_config["policy_maker_source"] = inspect.getsource(tune_config["policy_maker"])
    tune_config["lr_inner"] = tune_config["lr_inner"] * tune_config["base_lr"]
    tune_config["lr_outer"] = tune_config["lr_outer"] * tune_config["base_lr"]
    tune_config["lr_value"] = tune_config["lr_value"] * tune_config["base_lr"]
    tune_config["lr_om"] = tune_config["lr_om"] * tune_config["base_lr"]

    return tune_config


def main(debug):
    exp_name, _ = log.log_in_current_day_dir(f"LOLA_DICE")

    tune_hparams = {
        "debug": debug,
        "exp_name": exp_name,

        # "env_name": "IPD",
        # "env_name": "IMP",
        "env_name": "CoinGame",
        # "env_name": "AsymCoinGame",

        "gamma": None,
        "trace_length": None,

        "epochs": 0.2 if debug else 200,
        "lr_inner": .1,
        "lr_outer": .2,
        "lr_value": .1,
        "lr_om": .1,
        "inner_asymm": True,
        "n_agents": 2,
        "n_inner_steps": 1 if debug else 2,
        "batch_size": 4 if debug else 64,
        "value_batch_size": 16,
        "value_epochs": 0,
        "om_batch_size": 16,
        "om_epochs": 0,
        "grid_size": 3,
        "use_baseline": False,
        "use_dice": True,
        "use_opp_modeling": False,

        "seed": 1 if debug else tune.grid_search([1, 2, 3, 4, 5]),
    }

    tune_config = get_tune_config(tune_hparams)

    ray.init(num_cpus=os.cpu_count(), num_gpus=0)
    tune_analysis = tune.run(lola_training, name=tune_hparams["exp_name"], config=tune_config)
    ray.shutdown()

    return tune_analysis


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
