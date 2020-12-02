from gym.spaces import Discrete
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.stochastic_sampling import StochasticSampling
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.framework import get_variable
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule
from typing import Union


class SoftQSchedule(StochasticSampling):
    """Special case of StochasticSampling w/ Categorical and temperature param.

    Returns a stochastic sample from a Categorical parameterized by the model
    output divided by the temperature. Returns the argmax iff explore=False.
    """

    def __init__(self, action_space, *, framework,
                 initial_temperature=1.0, final_temperature=0.0,
                 temperature_timesteps=int(1e5),
                 temperature_schedule=None, **kwargs):
        """Initializes a SoftQ Exploration object.

        Args:
            action_space (Space): The gym action space used by the environment.
            temperature (Schedule): The temperature to divide model outputs by
                before creating the Categorical distribution to sample from.
            framework (str): One of None, "tf", "torch".
            temperature_schedule (Optional[Schedule]): An optional Schedule object
                to use (instead of constructing one from the given parameters).
        """
        assert isinstance(action_space, Discrete)
        super().__init__(action_space, framework=framework, **kwargs)

        self.temperature_schedule = \
            from_config(Schedule, temperature_schedule, framework=framework) or \
            PiecewiseSchedule(
                endpoints=[
                    (0, initial_temperature), (temperature_timesteps, final_temperature)],
                outside_value=final_temperature,
                framework=self.framework)

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            0, framework=framework, tf_name="timestep")
        self.temperature = self.temperature_schedule(self.last_timestep)

    @override(StochasticSampling)
    def get_exploration_action(self,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        cls = type(action_distribution)
        assert cls in [Categorical, TorchCategorical]

        self.last_timestep = timestep
        # TODO This step changes the Q value, even when we are not exploring, create an issue
        # Quick correction
        # TODO this correction breaks the LE algo?
        if explore:
            self.temperature = self.temperature_schedule(timestep if timestep is not None else self.last_timestep)
        else:
            self.temperature = 1.0

        # Re-create the action distribution with the correct temperature applied.
        dist = cls(
            action_distribution.inputs,
            self.model,
            temperature=self.temperature)
        # Delegate to super method.
        return super().get_exploration_action(action_distribution=dist, timestep=timestep, explore=explore)
