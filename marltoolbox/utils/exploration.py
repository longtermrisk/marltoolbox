import logging

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
import torch

logger = logging.getLogger(__name__)

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




class SoftQScheduleWtClustering(SoftQSchedule):

    def __init__(self,
                 action_space,
                 *,
                 framework,
                 clustering_distance=0.5,
                 **kwargs):

        super().__init__(action_space, framework=framework, **kwargs)

        self.clustering_distance = clustering_distance

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
        if explore:
            self.temperature = self.temperature_schedule(timestep if timestep is not None else self.last_timestep)
        else:
            self.temperature = 1.0

        self.clusterize(action_distribution.inputs)

        # Re-create the action distribution with the correct temperature applied.
        dist = cls(
            action_distribution.inputs,
            self.model,
            temperature=self.temperature)

        # Delegate to super method.
        return super().get_exploration_action(action_distribution=dist, timestep=timestep, explore=explore)

    def clusterize(self, action_distrib_inputs):
        logger.debug(f"before clustering action_distribution.inputs"
                     f" {action_distrib_inputs}")
        for batch_idx in range(len(action_distrib_inputs)):
            action_distrib_inputs[batch_idx, ...] = \
                self.clusterize_one_sample(
                    action_distrib_inputs[batch_idx, ...])
        logger.debug(f"after clustering action_distribution.inputs"
                     f" {action_distrib_inputs}")

    def clusterize_one_sample(self, one_action_distrib_input):
        return clusterize_by_distance(
                one_action_distrib_input.squeeze(dim=0),
                self.clustering_distance).unsqueeze(dim=0)

def group_cluster(key, clusters, group=None):
    if group is None:
        group = []
    if key in clusters.keys() and key not in group:
        group.append(key)
        for key_added in clusters[key]:
            group = group_cluster(key_added, clusters, group)
    return group

def clusterize_by_distance(pdparam, boltzmann_cluster_threshold):
    assert pdparam.dim() == 1, f"need pdparam.dim() == 1: pdparam.shape {pdparam.shape}"

    clusters = {}
    for i, p_1 in enumerate(pdparam.tolist()):
        for j, p_2 in enumerate(pdparam.tolist()):
            if i != j and abs(p_1 - p_2) < boltzmann_cluster_threshold:
                if i in clusters.keys():
                    clusters[i].append(j)
                else:
                    clusters[i] = [j]
    clustered_pdparam = pdparam.clone()
    for i, p_1 in enumerate(clustered_pdparam.tolist()):
        if i in clusters.keys():
            cluster_i = list(set(group_cluster(i, clusters)))
            clustered_pdparam[i] = pdparam[cluster_i].clone().mean()

    return clustered_pdparam

