from marltoolbox.utils.miscellaneous import move_to_key, logger


def extract_metric_values_per_trials(
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
