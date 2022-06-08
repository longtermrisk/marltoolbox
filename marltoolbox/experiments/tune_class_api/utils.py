import logging
import os

from marltoolbox.envs import (
    matrix_sequential_social_dilemma,
)
from marltoolbox.scripts import aggregate_and_plot_tensorboard_data
from marltoolbox.utils import (
    cross_play,
    log,
    miscellaneous,
)
from marltoolbox.utils.plot import PlotConfig
from ray import tune

logger = logging.getLogger(__name__)


def get_hyperparameters(debug, train_n_replicates=None, env=None):
    """Get hyperparameters for LOLA-Exact for matrix games"""

    if train_n_replicates is None:
        train_n_replicates = 2 if debug else int(3 * 1)
    seeds = miscellaneous.get_random_seeds(train_n_replicates)

    exp_name, _ = log.log_in_current_day_dir("LOLA_Exact")

    hparams = {
        "debug": debug,
        "load_plot_data": None,
        # Example "load_plot_data": ".../SelfAndCrossPlay_save.p",
        "exp_name": exp_name,
        "classify_into_welfare_fn": True,
        "train_n_replicates": train_n_replicates,
        "wandb": {
            "project": "LOLA_Exact",
            "group": exp_name,
            "api_key_file": os.path.join(
                os.path.dirname(__file__), "../../../api_key_wandb"
            ),
        },
        # "env_name": "IPD" if env is None else env,
        # "env_name": "IMP" if env is None else env,
        "env_name": "IteratedAsymBoS" if env is None else env,
        "num_episodes": 5 if debug else 50,
        "trace_length": 5 if debug else 200,
        "re_init_every_n_epi": 1,
        # "num_episodes": 5 if debug else 50 * 200,
        # "trace_length": 1,
        # "re_init_every_n_epi": 50,
        "simple_net": True,
        "corrections": True,
        "pseudo": False,
        "num_hidden": 32,
        "reg": 0.0,
        "lr": 1.0,
        "lr_correction": 1.0,
        "gamma": 0.96,
        "seed": tune.grid_search(seeds),
        "metric": "ret1",
        "with_linear_LR_decay_to_zero": False,
        "clip_update": None,
        # "with_linear_LR_decay_to_zero": True,
        # "clip_update": 0.1,
        # "lr": 0.001,
        "plot_keys": aggregate_and_plot_tensorboard_data.PLOT_KEYS + ["ret"],
        "plot_assemblage_tags": aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
        + [("ret",)],
        "x_limits": (-0.1, 4.1),
        "y_limits": (-0.1, 4.1),
    }

    hparams["plot_axis_scale_multipliers"] = (
        1 / hparams["trace_length"],
        1 / hparams["trace_length"],
    )
    return hparams


FAILURE = "failures"
EGALITARIAN = "egalitarian"
UTILITARIAN = "utilitarian"


def classify_into_welfare_based_on_rewards(reward_player_1, reward_player_2):

    ratio = reward_player_1 / reward_player_2
    if ratio < 1.5:
        return EGALITARIAN
    elif ratio < 2.5:
        return MIXED
    else:
        return UTILITARIAN


def lola_pg_classify_fn(
    pick_own_player_1, pick_own_player_2, hp, reward_player_1, reward_player_2
):
    if reward_player_2 != 0.0 and reward_player_1 != 0.0:
        if hp["env_name"] == "VectorizedSSDMixedMotiveCoinGame":
            ratio = reward_player_2 / reward_player_1
        else:
            ratio = max(
                reward_player_1 / reward_player_2,
                reward_player_2 / reward_player_1,
            )
        if ratio > 1.2:
            return UTILITARIAN
    return EGALITARIAN


def _evaluate_self_and_cross_perf(
    rllib_hp,
    rllib_config_eval,
    policies_to_load,
    trainable_class,
    stop,
    env_config,
    experiment_analysis_per_welfare,
    n_cross_play_per_checkpoint=None,
):
    exp_name = os.path.join(rllib_hp["exp_name"], "eval")
    evaluator = cross_play.evaluator.SelfAndCrossPlayEvaluator(
        exp_name=exp_name,
        local_mode=rllib_hp["debug"],
    )
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=rllib_config_eval,
        stop_config=stop,
        policies_to_load_from_checkpoint=policies_to_load,
        experiment_analysis_per_welfare=experiment_analysis_per_welfare,
        tune_trainer_class=trainable_class,
        n_cross_play_per_checkpoint=min(5, rllib_hp["train_n_replicates"] - 1)
        if n_cross_play_per_checkpoint is None
        else n_cross_play_per_checkpoint,
        to_load_path=rllib_hp["load_plot_data"],
    )

    if issubclass(
        rllib_hp["env_class"],
        matrix_sequential_social_dilemma.MatrixSequentialSocialDilemma,
    ):
        background_area_coord = rllib_hp["env_class"].PAYOFF_MATRIX
    else:
        background_area_coord = None

    plot_config = PlotConfig(
        xlim=rllib_hp["x_limits"],
        ylim=rllib_hp["y_limits"],
        markersize=5,
        jitter=rllib_hp["jitter"],
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        plot_max_n_points=rllib_hp["train_n_replicates"],
        x_scale_multiplier=rllib_hp["scale_multipliers"][0],
        y_scale_multiplier=rllib_hp["scale_multipliers"][1],
        background_area_coord=background_area_coord,
    )
    evaluator.plot_results(
        analysis_metrics_per_mode,
        plot_config=plot_config,
        x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}",
    )
