import os

import ray
from ray import tune
from ray.rllib.agents.dqn import R2D2Trainer

from marltoolbox.algos import augmented_dqn
from marltoolbox.envs import matrix_sequential_social_dilemma
from marltoolbox.examples.rllib_api import pg_ipd
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import log, miscellaneous


def main(debug):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("R2D2_IPD")

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)

    rllib_config, stop_config = pg_ipd.get_rllib_config(seeds, debug)
    rllib_config, stop_config = _adapt_configs_for_r2d2(
        rllib_config, stop_config, debug
    )

    tune_analysis = tune.run(
        R2D2Trainer,
        config=rllib_config,
        stop=stop_config,
        checkpoint_at_end=True,
        name=exp_name,
        log_to_file=True,
    )

    if not debug:
        _plot_log_aggregates(exp_name)

    ray.shutdown()
    return tune_analysis


def _adapt_configs_for_r2d2(rllib_config, stop_config, debug):
    rllib_config["model"] = {"use_lstm": True}
    stop_config["episodes_total"] = 10 if debug else 600

    return rllib_config, stop_config


def _plot_log_aggregates(exp_name):
    plot_keys = (
        aggregate_and_plot_tensorboard_data.PLOT_KEYS
        + matrix_sequential_social_dilemma.PLOT_KEYS
        + augmented_dqn.PL
    )
    plot_assemble_tags_in_one_plot = (
        aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
        + matrix_sequential_social_dilemma.PLOT_ASSEMBLAGE_TAGS
    )
    aggregate_and_plot_tensorboard_data.add_summary_plots(
        main_path=os.path.join("~/ray_results/", exp_name),
        plot_keys=plot_keys,
        plot_assemble_tags_in_one_plot=plot_assemble_tags_in_one_plot,
    )


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
