import copy

import os
import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_stats, DQNTorchPolicy

torch, nn = try_import_torch()

from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma
from marltoolbox.envs.coin_game import CoinGame
from marltoolbox.algos import amTFT
from marltoolbox.utils import experimenting, restore, exploration, log


def get_config(hparams):
    stop = {
        "episodes_total": hparams["n_epi"],  # 4000 steps in 200 epi
    }

    policyClass = amTFT.amTFTTorchPolicy

    if hparams["env"] == IteratedPrisonersDilemma:
        env_config = {
            "players_ids": ["player_row", "player_col"],
            "max_steps": hparams["n_steps_per_epi"],
            "reward_randomness": 0.1,
            "get_additional_info": True,
        }

        amTFT_config_update = merge_dicts(
            amTFT.DEFAULT_CONFIG_UPDATE,
            {
                # Set to True to train the nested policies and to False to use them
                "working_state": "train_coop",
            }
        )
    elif hparams["env"] == CoinGame:
        env_config = {
            "players_ids": ["player_red", "player_blue"],
            "max_steps": hparams["n_steps_per_epi"],
            "batch_size": 1,
            "grid_size": 3,
            "get_additional_info": True,
        }

        def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
            return torch.optim.SGD(policy.q_func_vars, lr=policy.cur_lr, momentum=config["sgd_momentum"])

        amTFT_config_update = merge_dicts(
            amTFT.DEFAULT_CONFIG_UPDATE,
            {
                # Set to True to train the nested policies and to False to use them
                "working_state": "train_coop",

                "sgd_momentum": 0.9,
                'nested_policies': [
                    {"Policy_class":
                         DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats),
                                                     optimizer_fn=sgd_optimizer_dqn),
                     "config_update": {}},
                    {"Policy_class":
                         DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats),
                                                     optimizer_fn=sgd_optimizer_dqn),
                     "config_update": {}},
                ]
            }
        )
    else:
        raise ValueError



    # TODO remove the useless hyper-parameters
    trainer_config_update = {
        "env": hparams["env"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    # The default policy is DQN defined in DQNTrainer but we overwrite it to use the LE policy
                    policyClass,
                    hparams["env"](env_config).OBSERVATION_SPACE,
                    hparams["env"].ACTION_SPACE,
                    copy.deepcopy(amTFT_config_update)
                ),
                env_config["players_ids"][1]: (
                    policyClass,
                    hparams["env"](env_config).OBSERVATION_SPACE,
                    hparams["env"].ACTION_SPACE,
                    copy.deepcopy(amTFT_config_update)
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },

        # === DQN Models ===
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": hparams["n_steps_per_epi"],
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": hparams["n_steps_per_epi"],
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(hparams["n_steps_per_epi"] * hparams["n_epi"]),
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [4],
        # Whether to use double dqn
        "double_q": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        "model": {
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": [4, 2],
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "relu",
        },

        "gamma": 0.5,
        "min_iter_time_s": 0.33,
        # Can't restaure stuff with search
        "seed": tune.grid_search([1, 2, 3, 4]),
        # "seed": 1,

        # "evaluation_num_episodes": 100,
        # "evaluation_interval": hparams["n_epi"],

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": hparams["base_lr"],
        # Learning rate schedule
        "lr_schedule": [(0, hparams["base_lr"]),
                        (int(hparams["n_steps_per_epi"] * hparams["n_epi"]), hparams["base_lr"] / 1e9)],
        # Adam epsilon hyper parameter
        # "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": 1,
        # How many steps of the model to sample before learning starts.
        "learning_starts": int(hparams["n_steps_per_epi"] * hparams["bs_epi_mul"]),
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": hparams["n_steps_per_epi"],
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": int(hparams["n_steps_per_epi"] * hparams["bs_epi_mul"]),

        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": exploration.SoftQSchedule,
            # Add constructor kwargs here (if any).
            "temperature_schedule": PiecewiseSchedule(
                endpoints=[
                    (0, 1.0), (int(hparams["n_steps_per_epi"] * hparams["n_epi"] * 0.75), 0.1)],
                outside_value=0.1,
                framework="torch")
        },

        # General config
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # LE supports only 1 worker only otherwise it would be mixing several opponents trajectories
        "num_workers": 0,
        # LE supports only 1 env per worker only otherwise several episodes would be played at the same time
        "num_envs_per_worker": 1,

        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
        # for more usage information.
        # "callbacks": DefaultCallbacks,
        "callbacks": amTFT.amTFTCallBacks,

    }

    if hparams["env"] == CoinGame:
        trainer_config_update["model"] = {
            "dim": env_config["grid_size"],
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]]  # [Channel, [Kernel, Kernel], Stride]]
        }

    return stop, env_config, trainer_config_update


def train(trainer_config_update, stop):
    # Train internal coop policy
    for policy_id in trainer_config_update["multiagent"]["policies"].keys():
        trainer_config_update["multiagent"]["policies"][policy_id][3]["working_state"] = "train_coop"
    results = ray.tune.run(amTFT.amTFTTrainer, config=trainer_config_update,
                           stop=stop, verbose=1, name="amTFT/amTFT",
                           checkpoint_at_end=True,
                           metric="episode_reward_mean", mode="max")

    # Train internal selfish policy
    for policy_id in trainer_config_update["multiagent"]["policies"].keys():
        trainer_config_update["multiagent"]["policies"][policy_id][3]["working_state"] = "train_selfish"
        trainer_config_update["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = \
            results.best_checkpoint
    amTFTTrainerTrainSelfish = restore.prepare_trainer_to_load_checkpoints(amTFT.amTFTTrainer)
    results = ray.tune.run(amTFTTrainerTrainSelfish, config=trainer_config_update,
                           stop=stop, verbose=1, name="amTFT/amTFT",
                           checkpoint_at_end=True,
                           # with restore it is not working easily
                           # since it reload the global_timestep too (and other undesired stuff maybe)
                           # restore=results.best_checkpoint,
                           metric="episode_reward_mean", mode="max")
    return results


def evaluate_same_and_cross_perf(trainer_config_update, results, hparams, env_config, stop):
    # Evaluate
    trainer_config_update["explore"] = False
    trainer_config_update["seed"] = 1000
    for policy_id in trainer_config_update["multiagent"]["policies"].keys():
        trainer_config_update["multiagent"]["policies"][policy_id][3]["working_state"] = "eval_amtft"
    policies_to_load = copy.deepcopy(env_config["players_ids"])
    if not hparams["self_play"]:
        naive_player_id = env_config["players_ids"][-1]
        trainer_config_update["multiagent"]["policies"][naive_player_id][3]["working_state"] = "eval_naive_selfish"
    experimenting.SameAndCrossPlay(TrainerClass=amTFT.amTFTTrainer,
                                   metrics=((f"policy_reward_mean/{env_config['players_ids'][0]}",
                                             f"policy_reward_mean/{env_config['players_ids'][1]}"),),
                                   extract_checkpoints_from_results=results,
                                   evaluation_config=trainer_config_update,
                                   stop_config=stop,
                                   exp_name="amTFT/amTFT",
                                   n_same_play_per_checkpoint=1,
                                   n_cross_play_per_checkpoint=1,
                                   policies_to_train=["None"],
                                   policies_to_load=policies_to_load
                                   )


# TODO update the unique rollout worker after every episode
# TODO check than no bug arise from the fact that there is 2 policies
#  (one used to produce samples in the rolloutworker and one used to train the models)
if __name__ == "__main__":
    hparams = {
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,

        # "self_play": True,
        "self_play": False,

        "env": IteratedPrisonersDilemma,
        # "env": CoinGame
    }
    if hparams["env"] == IteratedPrisonersDilemma:
        hparams["n_epi"] = 400
        hparams["base_lr"] = 0.04
    elif hparams["env"] == CoinGame:
        hparams["n_epi"] = 4000
        hparams["base_lr"] = 0.008
    else:
        raise NotImplementedError(f'hparams["env"]: {hparams["env"]}')

    ray.init(num_cpus=4, num_gpus=0)
    stop, env_config, trainer_config_update = get_config(hparams)
    results = train(trainer_config_update, stop)
    hparams["n_epi"] = 20
    hparams["n_steps_per_epi"] = 200
    stop, env_config, trainer_config_update = get_config(hparams)
    evaluate_same_and_cross_perf(trainer_config_update, results, hparams, env_config, stop)
    ray.shutdown()
