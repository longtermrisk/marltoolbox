from collections.abc import Callable

from ray import tune
from ray.rllib.utils import PiecewiseSchedule

from marltoolbox.utils.miscellaneous import move_to_key


def get_temp_scheduler() -> Callable:
    """
    Use the hyperparameter 'temperature_steps_config' stored inside the
    env_config dict since there are no control done over the keys of this
    dictionary.
    This hyperparameter is a list of tuples List[Tuple]. Each tuple define
    one step in the scheduler.

    :return: an easily customizable temperature scheduler
    """
    return _configurable_linear_scheduler("temperature_steps_config")


def get_lr_scheduler() -> Callable:
    """
    Use the hyperparameter 'lr_steps_config' stored inside the
    env_config dict since there are no control done over the keys of this
    dictionary.
    This hyperparameter is a list of tuples List[Tuple]. Each tuple define
    one step in the scheduler.

    :return: an easily customizable temperature scheduler
    """
    return _configurable_linear_scheduler(
        "lr_steps_config", second_term_key="lr"
    )


def _configurable_linear_scheduler(config_key, second_term_key: str = None):
    return tune.sample_from(
        lambda spec: PiecewiseSchedule(
            endpoints=[
                (
                    int(
                        spec.config["env_config"]["max_steps"]
                        * spec.stop["episodes_total"]
                        * step_config[0]
                    ),
                    step_config[1],
                )
                if second_term_key is None
                else (
                    int(
                        spec.config["env_config"]["max_steps"]
                        * spec.stop["episodes_total"]
                        * step_config[0]
                    ),
                    step_config[1]
                    * move_to_key(spec.config, second_term_key)[2],
                )
                for step_config in spec.config["env_config"][config_key]
            ],
            outside_value=spec.config["env_config"][config_key][-1][1],
            framework="torch",
        )
    )
