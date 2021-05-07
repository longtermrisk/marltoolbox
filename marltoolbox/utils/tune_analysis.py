from marltoolbox.utils.miscellaneous import move_to_key, logger


def extract_value_from_last_training_iteration_for_each_trials(
    tune_analysis,
    metric="episode_reward_mean",
):
    metric_values = []
    for trial in tune_analysis.trials:
        last_results = trial.last_result
        _, _, value, found = move_to_key(last_results, key=metric)
        assert (
            found
        ), f"metric: {metric} not found in last_results: {last_results}"
        metric_values.append(value)
    return metric_values


def extract_metrics_for_each_trials(
    tune_analysis,
    metric="episode_reward_mean",
    metric_mode="avg",
):
    metric_values = []
    for trial in tune_analysis.trials:
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


def extract_config_values_from_tune_analysis(tune_experiment_analysis, key):
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
    experiement_analysis,
    metric,
    metric_threshold: float,
    metric_mode="last-5-avg",
    threshold_mode=ABOVE,
):
    """
    Filter trials of an ExperimentAnalysis

    :param experiement_analysis:
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
    print(
        "Before trial filtering:", len(experiement_analysis.trials), "trials"
    )
    trials_filtered = []
    print(
        "metric_threshold", metric_threshold, "threshold_mode", threshold_mode
    )
    for trial_idx, trial in enumerate(experiement_analysis.trials):
        available_metrics = trial.metric_analysis
        metric_value = available_metrics[metric][metric_mode]
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
    experiement_analysis.trials = trials_filtered
    print("After trial filtering:", len(experiement_analysis.trials), "trials")
    return experiement_analysis
