import copy
import os

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

from marltoolbox.algos import amTFT
from marltoolbox.scripts.aggregate_and_plot_tensorboard_data import (
    add_summary_plots,
)
from marltoolbox.utils import miscellaneous, restore


def train_amtft(
    stop_config,
    rllib_config,
    name,
    do_not_load=[],
    TrainerClass=DQNTrainer,
    plot_keys=[],
    plot_assemblage_tags=[],
    debug=False,
    **kwargs,
):
    """
    Train a pair of similar amTFT policies in two steps.

    First training the selfis policies (nested inside the amTFT policies) and
    copying the weigths from one
    amTFT policy to the other. This allows the amTFT polciies to model the
    opponent policy.

    Second do the same but for the cooperative policies nested inside the
    amTFT policies.

    At the end, each amTFT policies contains 4 trained policies
    (own cooperative, own selfish, opponent cooperative, opponent selfish).

    :param stop_config: arg for ray.tune.run
    :param rllib_config: arg for ray.tune.run
    :param name: arg for ray.tune.run
    :param do_not_load: useless?
    :param TrainerClass: arg for ray.tune.run
    :param plot_keys: arg for add_summary_plots
    :param plot_assemblage_tags: arg for add_summary_plots
    :param debug: debug mode
    :param kwargs: kwargs for ray.tune.run
    :return: tune_analysis containing the checkpoints of the pair of amTFT
    policies
    """
    selfish_name = os.path.join(name, "selfish")
    tune_analysis_selfish_policies = _train_selfish_policies_inside_amtft(
        stop_config, rllib_config, selfish_name, TrainerClass, **kwargs
    )
    plot_keys, plot_assemblage_tags = _get_plot_keys(
        plot_keys, plot_assemblage_tags
    )
    if not debug:
        add_summary_plots(
            main_path=os.path.join("~/ray_results/", selfish_name),
            plot_keys=plot_keys,
            plot_assemble_tags_in_one_plot=plot_assemblage_tags,
        )

    seed_to_selfish_checkpoints = _extract_selfish_policies_checkpoints(
        tune_analysis_selfish_policies
    )
    rllib_config = _modify_config_to_load_selfish_policies_in_amtft(
        rllib_config, do_not_load, seed_to_selfish_checkpoints
    )

    coop_name = os.path.join(name, "coop")
    tune_analysis_amTFT_policies = _train_cooperative_policies_inside_amtft(
        stop_config, rllib_config, coop_name, TrainerClass, **kwargs
    )
    if not debug:
        add_summary_plots(
            main_path=os.path.join("~/ray_results/", coop_name),
            plot_keys=plot_keys,
            plot_assemble_tags_in_one_plot=plot_assemblage_tags,
        )
    return tune_analysis_amTFT_policies


def _train_selfish_policies_inside_amtft(
    stop_config, rllib_config, name, trainer_class, **kwargs
):
    rllib_config = copy.deepcopy(rllib_config)
    stop_config = copy.deepcopy(stop_config)

    for policy_id in rllib_config["multiagent"]["policies"].keys():
        rllib_config["multiagent"]["policies"][policy_id][3][
            "working_state"
        ] = "train_selfish"
    print("==============================================")
    print("amTFT starting to train the selfish policy")
    tune_analysis_selfish_policies = ray.tune.run(
        trainer_class,
        config=rllib_config,
        stop=stop_config,
        name=name,
        checkpoint_at_end=True,
        metric="episode_reward_mean",
        mode="max",
        **kwargs,
    )
    return tune_analysis_selfish_policies


def _get_plot_keys(plot_keys, plot_assemblage_tags):
    plot_keys = amTFT.PLOT_KEYS + plot_keys
    plot_assemble_tags_in_one_plot = (
        amTFT.PLOT_ASSEMBLAGE_TAGS + plot_assemblage_tags
    )
    return plot_keys, plot_assemble_tags_in_one_plot


def _extract_selfish_policies_checkpoints(tune_analysis_selfish_policies):
    checkpoints = miscellaneous.extract_checkpoints(
        tune_analysis_selfish_policies
    )
    seeds = miscellaneous.extract_config_values_from_tune_analysis(
        tune_analysis_selfish_policies, "seed"
    )
    seed_to_checkpoint = {}
    for seed, checkpoint in zip(seeds, checkpoints):
        seed_to_checkpoint[seed] = checkpoint
    return seed_to_checkpoint


def _modify_config_to_load_selfish_policies_in_amtft(
    rllib_config, do_not_load, seed_to_checkpoint
):
    for policy_id in rllib_config["multiagent"]["policies"].keys():
        if policy_id not in do_not_load:
            policy_config_dict = rllib_config["multiagent"]["policies"][
                policy_id
            ][3]
            policy_config_dict[restore.LOAD_FROM_CONFIG_KEY] = (
                miscellaneous.seed_to_checkpoint(seed_to_checkpoint),
                policy_id,
            )
    return rllib_config


def _train_cooperative_policies_inside_amtft(
    stop_config, rllib_config, name, trainer_class, **kwargs
):
    rllib_config = copy.deepcopy(rllib_config)
    stop_config = copy.deepcopy(stop_config)
    for policy_id in rllib_config["multiagent"]["policies"].keys():
        policy_config_dict = rllib_config["multiagent"]["policies"][policy_id][
            3
        ]
        policy_config_dict["working_state"] = "train_coop"

    print("==============================================")
    print("amTFT starting to train the cooperative policy")
    tune_analysis_amTFT_policies = ray.tune.run(
        trainer_class,
        config=rllib_config,
        stop=stop_config,
        name=name,
        checkpoint_at_end=True,
        metric="episode_reward_mean",
        mode="max",
        **kwargs,
    )
    return tune_analysis_amTFT_policies
