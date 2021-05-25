import logging

from ray.tune import Trainable
from ray.tune import register_trainable
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.checkpoint_manager import Checkpoint
from ray.tune.trial import Trial

from marltoolbox.utils.miscellaneous import (
    move_to_key,
    _get_experiment_state_file_path,
)

logger = logging.getLogger(__name__)


def extract_value_from_last_training_iteration_for_each_trials(
    experiment_analysis,
    metric="episode_reward_mean",
):
    metric_values = []
    for trial in experiment_analysis.trials:
        last_results = trial.last_result
        _, _, value, found = move_to_key(last_results, key=metric)
        assert (
            found
        ), f"metric: {metric} not found in last_results: {last_results}"
        metric_values.append(value)
    return metric_values


def extract_metrics_for_each_trials(
    experiment_analysis,
    metric="episode_reward_mean",
    metric_mode="avg",
):
    metric_values = []
    for trial in experiment_analysis.trials:
        metric_values.append(trial.metric_analysis[metric][metric_mode])
    return metric_values


def check_learning_achieved(
    tune_results,
    metric="episode_reward_mean",
    trial_idx=0,
    max_: float = None,
    min_: float = None,
    equal_: float = None,
):
    assert max_ is not None or min_ is not None or equal_ is not None

    last_results = tune_results.trials[trial_idx].last_result
    _, _, value, found = move_to_key(last_results, key=metric)
    assert (
        found
    ), f"metric {metric} not found inside last_results {last_results}"

    msg = (
        f"Trial {trial_idx} achieved "
        f"{value}"
        f" on metric {metric}. This is a success if the value is below"
        f" {max_} or above {min_} or equal to {equal_}."
    )

    logger.info(msg)
    print(msg)
    if min_ is not None:
        assert value >= min_, f"value {value} must be above min_ {min_}"
    if max_ is not None:
        assert value <= max_, f"value {value} must be below max_ {max_}"
    if equal_ is not None:
        assert value == equal_, (
            f"value {value} must be equal to equal_ " f"{equal_}"
        )


def extract_config_values_from_experiment_analysis(
    tune_experiment_analysis, key
):
    values = []
    for trial in tune_experiment_analysis.trials:
        dict_, k, current_value, found = move_to_key(trial.config, key)
        if found:
            values.append(current_value)
        else:
            values.append(None)
    return values


ABOVE = "above"
EQUAL = "equal"
BELOW = "below"
FILTERING_MODES = (ABOVE, EQUAL, BELOW)

RLLIB_METRICS_MODES = (
    "avg",
    "min",
    "max",
    "last",
    "last-5-avg",
    "last-10-avg",
)


def filter_trials(
    experiment_analysis,
    metric,
    metric_threshold: float,
    metric_mode="last-5-avg",
    threshold_mode=ABOVE,
):
    """
    Filter trials of an ExperimentAnalysis

    :param experiment_analysis:
    :param metric:
    :param metric_threshold:
    :param metric_mode:
    :param threshold_mode:
    :return:
    """
    assert threshold_mode in FILTERING_MODES, (
        f"threshold_mode {threshold_mode} " f"must be in {FILTERING_MODES}"
    )
    assert metric_mode in RLLIB_METRICS_MODES
    print("Before trial filtering:", len(experiment_analysis.trials), "trials")
    trials_filtered = []
    print(
        "metric_threshold", metric_threshold, "threshold_mode", threshold_mode
    )
    for trial_idx, trial in enumerate(experiment_analysis.trials):
        available_metrics = trial.metric_analysis
        try:
            metric_value = available_metrics[metric][metric_mode]
        except KeyError:
            raise KeyError(
                f"failed to read metric key:{metric} in "
                f"available_metrics:{available_metrics}"
            )
        print(
            f"trial_idx {trial_idx} "
            f"available_metrics[{metric}][{metric_mode}] "
            f"{metric_value}"
        )
        if threshold_mode == ABOVE and metric_value > metric_threshold:
            trials_filtered.append(trial)
        elif threshold_mode == EQUAL and metric_value == metric_threshold:
            trials_filtered.append(trial)
        elif threshold_mode == BELOW and metric_value < metric_threshold:
            trials_filtered.append(trial)
        else:
            print(f"filtering out trial {trial_idx}")

    experiment_analysis.trials = trials_filtered
    print("After trial filtering:", len(experiment_analysis.trials), "trials")
    return experiment_analysis


def filter_trials_wt_n_metrics(
    experiment_analysis,
    metrics: tuple,
    metric_thresholds: tuple,
    metric_modes: tuple,
    threshold_modes: tuple,
):
    for threshold_mode in threshold_modes:
        assert threshold_mode in FILTERING_MODES, (
            f"threshold_mode {threshold_mode} " f"must be in {FILTERING_MODES}"
        )
    for metric_mode in metric_modes:
        assert metric_mode in RLLIB_METRICS_MODES
    print("Before trial filtering:", len(experiment_analysis.trials), "trials")
    trials_filtered = []
    print(
        "metric_thresholds",
        metric_thresholds,
        "threshold_modes",
        threshold_modes,
    )
    for trial_idx, trial in enumerate(experiment_analysis.trials):
        keep = []
        for metric, metric_threshold, metric_mode, threshold_mode in zip(
            metrics, metric_thresholds, metric_modes, threshold_modes
        ):

            available_metrics = trial.metric_analysis
            try:
                metric_value = available_metrics[metric][metric_mode]
            except KeyError:
                raise KeyError(
                    f"failed to read metric key:{metric} in "
                    f"available_metrics:{available_metrics}"
                )
            print(
                f"trial_idx {trial_idx} "
                f"available_metrics[{metric}][{metric_mode}] "
                f"{metric_value}"
            )
            if threshold_mode == ABOVE and metric_value > metric_threshold:
                keep.append(True)
            elif threshold_mode == EQUAL and metric_value == metric_threshold:
                keep.append(True)
            elif threshold_mode == BELOW and metric_value < metric_threshold:
                keep.append(True)
            else:
                keep.append(False)
        # Logical and between metrics
        if all(keep):
            trials_filtered.append(trial)

    experiment_analysis.trials = trials_filtered
    print("After trial filtering:", len(experiment_analysis.trials), "trials")
    return experiment_analysis


def load_experiment_analysis_wt_ckpt_only(
    checkpoints_paths: list,
    result: dict = {"training_iteration": 1, "episode_reward_mean": 1},
    default_metric: "str" = "episode_reward_mean",
    default_mode: str = "max",
    n_dir_level_between_ckpt_and_exp_state=1,
):
    """Helper to re-create a fake ExperimentAnalysis only containing the
    checkpoints provided."""

    assert default_metric in result.keys()

    register_trainable("fake trial", Trainable)
    trials = []
    for one_checkpoint_path in checkpoints_paths:
        one_trial = Trial(trainable_name="fake trial")
        ckpt = Checkpoint(
            Checkpoint.PERSISTENT, value=one_checkpoint_path, result=result
        )
        one_trial.checkpoint_manager.on_checkpoint(ckpt)
        trials.append(one_trial)

    json_file_path = _get_experiment_state_file_path(
        checkpoints_paths[0],
        split_path_n_times=n_dir_level_between_ckpt_and_exp_state,
    )
    experiment_analysis = ExperimentAnalysis(
        experiment_checkpoint_path=json_file_path,
        trials=trials,
        default_mode=default_mode,
        default_metric=default_metric,
    )

    for trial in experiment_analysis.trials:
        assert len(trial.checkpoint_manager.best_checkpoints()) == 1

    return experiment_analysis


from collections import namedtuple

FakeExperimentAnalysis = namedtuple("FakeExperimentAnalysis", "trials")


def create_fake_experiment_analysis_wt_metrics_only(
    results: dict = ({"training_iteration": 1, "episode_reward_mean": 1},),
    default_metric: "str" = "episode_reward_mean",
    default_mode: str = "max",
):
    """Helper to re-create a fake ExperimentAnalysis only containing the
    checkpoints provided."""

    # assert default_metric in result.keys()

    register_trainable("fake trial", Trainable)
    trials = []
    for result in results:
        trial = Trial(trainable_name="fake trial")
        trial.update_last_result(result, terminate=True)
        trials.append(trial)

    # experiment_analysis = ExperimentAnalysis(
    #     experiment_checkpoint_path="fake_path",
    #     trials=trials,
    #     default_mode=default_mode,
    #     default_metric=default_metric,
    # )

    experiment_analysis = FakeExperimentAnalysis(trials=trials)

    return experiment_analysis
