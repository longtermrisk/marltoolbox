import torch
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from marltoolbox.algos.amTFT import base
from marltoolbox.algos.hierarchical import HierarchicalTorchPolicy


class StochasticPopulation(HierarchicalTorchPolicy, base.AmTFTReferenceClass):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)
        self.stochastic_population_policy = torch.distributions.Categorical(
            probs=self.config["sampling_policy_distribution"]
        )

    def on_episode_start(self, *args, **kwargs):
        self._select_algo_idx_to_use()
        if hasattr(self.algorithms[self.active_algo_idx], "on_episode_start"):
            self.algorithms[self.active_algo_idx].on_episode_start(
                *args,
                **kwargs,
            )

    def _select_algo_idx_to_use(self):
        policy_idx_selected = self.stochastic_population_policy.sample()
        self.active_algo_idx = policy_idx_selected
        self._to_log[
            "StochasticPopulation_active_algo_idx"
        ] = self.active_algo_idx

    @override(HierarchicalTorchPolicy)
    def _learn_on_batch(self, samples: SampleBatch):
        return self.algorithms[self.active_algo_idx]._learn_on_batch(samples)
