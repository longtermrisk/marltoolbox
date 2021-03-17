import copy
import os
import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

from marltoolbox.algos import amTFT
from marltoolbox.utils import miscellaneous, restore
from marltoolbox.scripts.aggregate_and_plot_tensorboard_data import \
    add_summary_plots

def train_amTFT(stop, config, name, do_not_load=[],
                TrainerClass=DQNTrainer,
                plot_keys=[],
                plot_assemblage_tags=[],
                debug=False,
                **kwargs):
    selfish_name = os.path.join(name, "selfish")
    tune_analysis_selfish_policies = _train_selfish_policies_inside_amTFT(
        stop, config, selfish_name, TrainerClass, **kwargs)
    plot_keys, plot_assemblage_tags = \
        _get_plot_keys(plot_keys, plot_assemblage_tags)
    if not debug:
        add_summary_plots(
            main_path=os.path.join("~/ray_results/", selfish_name),
            plot_keys=plot_keys,
            plot_assemble_tags_in_one_plot=plot_assemblage_tags,
        )

    seed_to_selfish_checkpoints = _extract_selfish_policies_checkpoints(tune_analysis_selfish_policies)
    config = _modify_config_to_load_selfish_policies_in_amTFT(config, do_not_load, seed_to_selfish_checkpoints)

    coop_name = os.path.join(name, "coop")
    tune_analysis_amTFT_policies = _train_cooperative_policies_inside_amTFT(
        stop, config, coop_name, TrainerClass, **kwargs)
    if not debug:
        add_summary_plots(
            main_path=os.path.join("~/ray_results/", coop_name),
            plot_keys=plot_keys,
            plot_assemble_tags_in_one_plot=plot_assemblage_tags,
        )
    return tune_analysis_amTFT_policies


def _train_selfish_policies_inside_amTFT(stop, config, name, TrainerClass, **kwargs):
    for policy_id in config["multiagent"]["policies"].keys():
        config["multiagent"]["policies"][policy_id][3]["working_state"] = "train_selfish"
    print("==============================================")
    print("amTFT starting to train the selfish policy")
    tune_analysis_selfish_policies = ray.tune.run(TrainerClass, config=config,
                                                  stop=stop, name=name,
                                                  checkpoint_at_end=True,
                                                  metric="episode_reward_mean", mode="max",
                                                  **kwargs)
    return tune_analysis_selfish_policies


def _get_plot_keys(plot_keys, plot_assemblage_tags):
    plot_keys = \
        amTFT.PLOT_KEYS + \
        plot_keys
    plot_assemble_tags_in_one_plot = \
        amTFT.PLOT_ASSEMBLAGE_TAGS + \
        plot_assemblage_tags
    return plot_keys, plot_assemble_tags_in_one_plot


def _extract_selfish_policies_checkpoints(tune_analysis_selfish_policies):
    checkpoints = miscellaneous.extract_checkpoints(tune_analysis_selfish_policies)
    seeds = miscellaneous.extract_config_values_from_tune_analysis(tune_analysis_selfish_policies, "seed")
    seed_to_checkpoint = {}
    for seed, checkpoint in zip(seeds, checkpoints):
        seed_to_checkpoint[seed] = checkpoint
    return seed_to_checkpoint


def _modify_config_to_load_selfish_policies_in_amTFT(config, do_not_load, seed_to_checkpoint):
    for policy_id in config["multiagent"]["policies"].keys():
        if policy_id not in do_not_load:
            config["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = (
                miscellaneous.seed_to_checkpoint(seed_to_checkpoint), policy_id
            )
    return config


def _train_cooperative_policies_inside_amTFT(stop, config, name, TrainerClass, **kwargs):
    for policy_id in config["multiagent"]["policies"].keys():
        config["multiagent"]["policies"][policy_id][3]["working_state"] = "train_coop"

    print("==============================================")
    print("amTFT starting to train the cooperative policy")
    tune_analysis_amTFT_policies = ray.tune.run(TrainerClass, config=config,
                                                stop=stop, name=name,
                                                checkpoint_at_end=True,
                                                metric="episode_reward_mean", mode="max",
                                                **kwargs)
    return tune_analysis_amTFT_policies
