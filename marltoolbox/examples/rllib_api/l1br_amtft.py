import copy
import os

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from marltoolbox.envs import matrix_sequential_social_dilemma
from marltoolbox.algos import amTFT, population
from marltoolbox.utils import log, \
    postprocessing, lvl1_best_response, miscellaneous
from marltoolbox.examples.rllib_api import amtft_various_env


def train_lvl0_population(hp):
    assert len(hp["welfare_functions"]) == 1
    lvl0_tune_analysis_per_welfare = amtft_various_env.train_for_each_welfare_function(hp)
    assert len(lvl0_tune_analysis_per_welfare) == 1
    tune_analysis_lvl0 = list(lvl0_tune_analysis_per_welfare.values())[0]
    return tune_analysis_lvl0


def train_lvl1_agents(hp_lvl1, tune_analysis_lvl0):
    print("================================================================================")
    print("Start training level 1 agents against populations of frozen level 0 amTFT agents")

    stop, env_config, rllib_config = amtft_various_env.get_rllib_config(hp_lvl1, hp_lvl1['welfare_functions'][0][0])
    checkpoints_lvl0 = miscellaneous.extract_checkpoints(tune_analysis_lvl0)
    rllib_config = modify_conf_for_lvl1_training(hp_lvl1, env_config, rllib_config, checkpoints_lvl0)

    tune_analysis_lvl1 = ray.tune.run(dqn.DQNTrainer, config=rllib_config,
                                      stop=stop, name=hp_lvl1["exp_name"],
                                      checkpoint_at_end=True,
                                      metric="episode_reward_mean", mode="max")
    return tune_analysis_lvl1


def modify_conf_for_lvl1_training(hp_lvl1, env_config, rllib_config_lvl1, lvl0_checkpoints):
    lvl0_policy_idx = 1
    lvl1_policy_idx = 0

    lvl0_policy_id = env_config["players_ids"][lvl0_policy_idx]
    lvl1_policy_id = env_config["players_ids"][lvl1_policy_idx]

    # Use a simple DQN as lvl1 agent (instead of amTFT with nested DQN)
    rllib_config_lvl1["multiagent"]["policies"][lvl1_policy_id] = (
        DQNTorchPolicy,
        hp_lvl1["env"](env_config).OBSERVATION_SPACE,
        hp_lvl1["env"].ACTION_SPACE,
        {}
    )

    rllib_config_lvl1["callbacks"] = amTFT.get_amTFTCallBacks(
        additionnal_callbacks=[log.get_logging_callbacks_class(),
                               postprocessing.OverwriteRewardWtWelfareCallback,
                               population.PopulationOfIdenticalAlgoCallBacks])

    l1br_configuration_helper = lvl1_best_response.L1BRConfigurationHelper(rllib_config_lvl1, lvl0_policy_id, lvl1_policy_id)
    l1br_configuration_helper.define_exp(
        use_n_lvl0_agents_in_each_population=hp_lvl1["n_seeds_lvl0"] // hp_lvl1["n_seeds_lvl1"],
        train_n_lvl1_agents=hp_lvl1["n_seeds_lvl1"],
        lvl0_checkpoints=lvl0_checkpoints)
    rllib_config_lvl1 = l1br_configuration_helper.prepare_config_for_lvl1_training()

    # rllib_config_lvl1["multiagent"]["policies"][lvl0_policy_id][3]["explore"] = False
    rllib_config_lvl1["multiagent"]["policies"][lvl0_policy_id][3]["working_state"] = "eval_amtft"
    return rllib_config_lvl1


def main(debug):
    exp_name, _ = log.log_in_current_day_dir("L1BR_amTFT")

    train_n_replicates = 4 if debug else 8
    pool_of_seeds = miscellaneous.get_random_seeds(train_n_replicates)
    hparams = {
        "debug": debug,
        "filter_utilitarian": False,

        "train_n_replicates": train_n_replicates,
        "seeds": pool_of_seeds,

        "exp_name": exp_name,
        "n_steps_per_epi": 20,
        "bs_epi_mul": 4,
        "welfare_functions": [(postprocessing.WELFARE_UTILITARIAN, "utilitarian")],

        "amTFTPolicy": amTFT.amTFTRolloutsTorchPolicy,
        "explore_during_evaluation": True,

        "n_seeds_lvl0": train_n_replicates,
        "n_seeds_lvl1": train_n_replicates//2,

        "gamma": 0.5,
        "lambda": 0.9,
        "alpha": 0.0,
        "beta": 1.0,

        "temperature_schedule": False,
        "debit_threshold": 4.0,
        "jitter": 0.05,
        "hiddens": [64],

        "env": matrix_sequential_social_dilemma.IteratedPrisonersDilemma,
        # "env": matrix_sequential_social_dilemma.IteratedAsymBoS,
        # "env": matrix_sequential_social_dilemma.IteratedAsymChicken,
        # "env": coin_game.CoinGame
        # "env": coin_game.AsymCoinGame

        # For training speed
        "min_iter_time_s": 0.0 if debug else 3.0,

        "overwrite_reward": True,
        "use_adam": False,
    }

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=hparams["debug"])

    hparams = amtft_various_env.modify_hyperparams_for_the_selected_env(hparams)
    lvl0_tune_analysis = train_lvl0_population(hp=hparams)
    tune_analysis_lvl1 = train_lvl1_agents(hp_lvl1=copy.deepcopy(hparams), tune_analysis_lvl0=lvl0_tune_analysis)
    print(tune_analysis_lvl1.results_df.columns)
    print(tune_analysis_lvl1.results_df.head())

    ray.shutdown()

if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
