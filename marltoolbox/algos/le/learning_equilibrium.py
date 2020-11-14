##########
# Code modified from: https://github.com/Manuscrit/SLM-Lab/blob/support_multi_agents_with_awareness/slm_lab/agent/algorithm/meta_algorithm/learning_equilibrium.py
# WORK IN PROGRESS!! This implementation is not working properly.
##########

import copy
import logging
import numbers
from collections import deque
from typing import List, Union, Optional, Dict, Tuple

import numpy as np
import torch
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType, AgentID

from marltoolbox.algos.supervised_learning import SPL_DEFAULT_CONFIG, SPLTorchPolicy

LE_DEFAULT_CONFIG_UPDATE = {
    "percentile_for_likelihood_test": 95,
    "punishment_time": 1,
    "min_coop_epi_after_punishing": 0,
    "defection_threshold": 0.0,
    "average_defection_value": True,
    "average_defection_value_len": 20,
    "use_last_step_for_search": True,
    "length_of_history": 200,
    "n_steps_in_bootstrap_replicates": 20,
    "n_bootstrap_replicates": 50,
    'nested_policies': [
        # Here the trainer need to be a DQNTrainer to provide the config for the 3 DQNTorchPolicy
        {"Policy_class": DQNTorchPolicy, "config_update": {}},
        {"Policy_class": DQNTorchPolicy, "config_update": {}},
        {"Policy_class": DQNTorchPolicy, "config_update": {}},
        {"Policy_class": SPLTorchPolicy, "config_update": copy.deepcopy(SPL_DEFAULT_CONFIG)},
    ],
}


# TODO make a parent class for nested/hierarchical algo?
class LE(rllib.policy.TorchPolicy):
    """
    Learning Tit-for-tat or Learning Equilibrium(LE)
    """
    EPSILON = 1e-12
    DEBUG = False

    COOP_POLICY_IDX = 0
    PUNITIVE_POLICY_IDX = 1
    COOP_OPP_POLICY_IDX = 2
    SPL_OPP_POLICY_IDX = 3

    WARMUP_LENGTH = 0
    INITIAL_ACTIVE_ALGO = COOP_POLICY_IDX

    UTILITARIAN_WELFARE_KEY = "utilitarian_welfare"
    OPPONENT_NEGATIVE_REWARD_KEY = "opponent_negative_reward"
    OPPONENT_ACTIONS_KEY = "opponent_actions"

    def __init__(self, observation_space, action_space, config, **kwargs):

        self.percentile_for_likelihood_test = config['percentile_for_likelihood_test']
        self.punishment_time = config['punishment_time']
        self.min_coop_epi_after_punishing = config['min_coop_epi_after_punishing']
        self.defection_threshold = config['defection_threshold']
        self.average_defection_value = config['average_defection_value'],
        self.average_defection_value_len = config['average_defection_value_len']
        self.use_last_step_for_search = config['use_last_step_for_search']
        self.length_of_history = config['length_of_history']
        self.n_steps_in_bootstrap_replicates = config['n_steps_in_bootstrap_replicates']
        self.n_bootstrap_replicates = config['n_bootstrap_replicates']

        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

        self.active_algo_idx = self.INITIAL_ACTIVE_ALGO

        self.algorithms = []
        for nested_config in config["nested_policies"]:
            updated_config = copy.deepcopy(config)
            updated_config.update(nested_config["config_update"])
            Policy = nested_config["Policy_class"]
            self.algorithms.append(Policy(observation_space, action_space, updated_config, **kwargs))

        assert len(self.algorithms) == 4, str(len(self.algorithms))
        self.opp_policy_from_supervised_learning = True

        self.n_steps_since_start = 0
        self.last_computed_w = None

        self.data_queue = deque(maxlen=self.length_of_history)

        # Defection
        self.detected_defection = False
        self.defection_metric = 0
        if not self.average_defection_value:
            self.average_defection_value_len = 1
        self.defection_carac_queue = deque(maxlen=self.average_defection_value_len)
        self.remaining_punishing_time = 0
        self.being_punished_by_LE = False

        # Logging
        self.to_log = {}
        log_len_in_steps = 100
        # for 1) Cooperative policy
        self.action_pd_coop = deque(maxlen=log_len_in_steps)
        # for 2) Punitive policy
        self.action_pd_punish = deque(maxlen=log_len_in_steps)
        # for 3) Estimated opponent cooperative policy
        self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
        # for 4) Estimated opponent policy
        self.action_pd_opp = deque(maxlen=log_len_in_steps)
        self.n_cooperation_steps_in_current_epi = 0
        self.n_punishment_steps_in_current_epi = 0

    @property
    def model(self):
        return self.algorithms[self.active_algo_idx].model

    @property
    def dist_class(self):
        return self.algorithms[self.active_algo_idx].dist_class

    @property
    def global_timestep(self):
        return self.algorithms[self.active_algo_idx].global_timestep

    @global_timestep.setter
    def global_timestep(self, value):
        for algo in self.algorithms:
            algo.global_timestep = value

    @property
    def action_space_struct(self):
        return self.algorithms[self.active_algo_idx].action_space_struct

    @property
    def update_target(self):
        def nested_update_target():
            for algo in self.algorithms:
                if "update_target" in algo.__dict__:
                    algo.update_target()

        return nested_update_target

    def compute_actions(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        assert len(obs_batch) == 1, "LE only works with one step at a time"
        state = obs_batch

        action, _, algo_info = self.algorithms[self.active_algo_idx].compute_actions(state)
        self.last_used_algo = self.active_algo_idx

        action_info_batch = {"punishing": [self.active_algo_idx == self.PUNITIVE_POLICY_IDX for _ in obs_batch]}

        return action, [], action_info_batch

    def learn_on_batch(self, samples: SampleBatch):
        learner_stats = {"learner_stats": {}}
        for i, algo in enumerate(self.algorithms):
            # TODO filter steps used to train
            #  (like when being punished => do not use these steps to infer the opponent coop policy)
            # TODO use the RLLib batch format directly
            samples_copy = copy.deepcopy(samples)
            if i == self.COOP_POLICY_IDX or i == self.COOP_OPP_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[self.UTILITARIAN_WELFARE_KEY])
            elif i == self.PUNITIVE_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[self.OPPONENT_NEGATIVE_REWARD_KEY])
            elif i == self.SPL_OPP_POLICY_IDX:
                samples_copy[samples_copy.ACTIONS] = np.array(samples_copy.data[self.OPPONENT_ACTIONS_KEY])

            learner_stats["learner_stats"][f"learner_stats_algo{i}"] = algo.learn_on_batch(samples_copy)

        learner_stats.update(self.to_log)
        self.to_log = {}
        return learner_stats

    def get_weights(self):
        return {i: algo.get_weights() for i, algo in enumerate(self.algorithms)}

    def set_weights(self, weights):
        for i, algo in enumerate(self.algorithms):
            algo.set_weights(weights[i])

    def optimizer(self
                  ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        """Custom the local PyTorch optimizer(s) to use.

        Returns:
            Union[List[torch.optim.Optimizer], torch.optim.Optimizer]:
                The local PyTorch optimizer(s) to use for this Policy.
        """
        raise NotImplementedError()

    def on_episode_step(self, opp_obs, opp_a, being_punished_by_LE):

        self.being_punished_by_LE = being_punished_by_LE

        if not self.being_punished_by_LE:
            # The opponent agent is not currently broadcasting "punishing"
            self.n_steps_since_start += 1
            self._put_log_likelihood_in_data_buffer(opp_obs, opp_a, self.data_queue)

        if self.remaining_punishing_time > 0:
            self.n_punishment_steps_in_current_epi += 1
        else:
            self.n_cooperation_steps_in_current_epi += 1

    def on_episode_end(self):

        if self.n_steps_since_start >= self.length_of_history + self.WARMUP_LENGTH:
            percentile_value = self._compare_log_likelihood_on_boostrapped_sequences(self.data_queue)
            self._update_defection_metric(epi_defection_metric=-percentile_value)

        if self.remaining_punishing_time <= - (self.min_coop_epi_after_punishing - 1):
            if self.defection_metric > self.defection_threshold:
                self.detected_defection = True

        # Averaged by episode
        self.to_log["coop_frac"] = (self.n_cooperation_steps_in_current_epi /
                                    (self.n_punishment_steps_in_current_epi + self.n_cooperation_steps_in_current_epi))
        self.n_cooperation_steps_in_current_epi = 0
        self.n_punishment_steps_in_current_epi = 0

        if self.remaining_punishing_time > - self.min_coop_epi_after_punishing:
            self.remaining_punishing_time -= 1

        # Switch from coop to punishment only at the start of epi
        if self.detected_defection and not self.being_punished_by_LE:
            self.active_algo_idx = self.PUNITIVE_POLICY_IDX
            self.remaining_punishing_time = self.punishment_time
            self.detected_defection = False
            print("DEFECTION DETECTED")

        if self.remaining_punishing_time <= 0:
            self.active_algo_idx = self.COOP_POLICY_IDX

    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[
                "Policy", SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        """Implements algorithm-specific trajectory postprocessing.

        This will be called on each trajectory fragment computed during policy
        evaluation. Each fragment is guaranteed to be only from one episode.

        Args:
            sample_batch (SampleBatch): batch of experiences for the policy,
                which will contain at most one episode trajectory.
            other_agent_batches (dict): In a multi-agent env, this contains a
                mapping of agent ids to (policy, agent_batch) tuples
                containing the policy and experiences of the other agents.
            episode (Optional[MultiAgentEpisode]): An optional multi-agent
                episode object to provide access to all of the
                internal episode state, which may be useful for model-based or
                multi-agent algorithms.

        Returns:
            SampleBatch: Postprocessed sample batch.
        """
        opponent_agent_ids = list(other_agent_batches.keys())
        assert len(opponent_agent_ids) == 1
        opponent_agent_id = opponent_agent_ids[0]
        opponent_agent_batch = other_agent_batches[opponent_agent_id][1]

        # For all experiences
        # Add the utilitarian welfare
        sample_batch.data[self.UTILITARIAN_WELFARE_KEY] = \
            [own_r + opp_r for own_r, opp_r in \
             zip(sample_batch[sample_batch.REWARDS], opponent_agent_batch[opponent_agent_batch.REWARDS])]
        # Add the opposite of the opponent reward
        sample_batch.data[self.OPPONENT_NEGATIVE_REWARD_KEY] = \
            [- opp_r for opp_r in \
             opponent_agent_batch[opponent_agent_batch.REWARDS]]
        # Add the opponent action
        sample_batch.data[self.OPPONENT_ACTIONS_KEY] = \
            opponent_agent_batch[opponent_agent_batch.ACTIONS]

        return sample_batch

    def _put_log_likelihood_in_data_buffer(self, s, a, data_queue, log=True):
        (log_likelihood_opponent_cooperating,
         self.opp_coop_a_prob_distrib) = (
            compute_s_a_log_likelihood(s=s,
                                       a=a,
                                       algo=self.algorithms[self.COOP_OPP_POLICY_IDX],
                                       no_grad=False,
                                       epsilon=self.EPSILON,
                                       debug=self.DEBUG))

        (log_likelihood_approximated_opponent,
         self.opp_spl_a_prob_distrib) = (
            compute_s_a_log_likelihood(
                s=s, a=a,
                algo=self.algorithms[self.SPL_OPP_POLICY_IDX],
                epsilon=self.EPSILON,
                debug=self.DEBUG)
        )

        data_queue.append([log_likelihood_opponent_cooperating,
                           log_likelihood_approximated_opponent])

    def _bootstrap_replicats(self, data_queue, last_step_is_mandatory):
        data_array = np.array(list(data_queue), dtype=np.object)
        maximum_idx = data_array.shape[0] - 1
        bstrap_idx = np.random.random_integers(0, high=maximum_idx,
                                               size=(self.n_bootstrap_replicates,
                                                     self.n_steps_in_bootstrap_replicates))
        if last_step_is_mandatory:
            # TODO only add it if it is not already present
            bstrap_idx[:, -1] = [maximum_idx for _ in range(self.n_bootstrap_replicates)]
        bstrap_replts_data = data_array[bstrap_idx]
        return bstrap_replts_data

    def _compare_log_likelihood_on_boostrapped_sequences(self, data_queue, log=True,
                                                         last_step_is_mandatory=False):

        bstrap_replts_data = self._bootstrap_replicats(data_queue, last_step_is_mandatory)

        # Sum log_likelihood over u steps
        log_lik_cooperate = bstrap_replts_data[:, :, 0].sum(axis=1)

        log_lik_defect = bstrap_replts_data[:, :, 1].sum(axis=1)

        # Defect if in more than 0.95 of the replicates, the actual policy is more likely than the simulated coop policy
        log_lik_check_coop = log_lik_cooperate - log_lik_defect
        assert len(log_lik_check_coop) == self.n_bootstrap_replicates
        percentile_value = np.percentile(log_lik_check_coop, self.percentile_for_likelihood_test,
                                         interpolation="linear")
        percentile_0_5_value = np.percentile(log_lik_check_coop, 50, interpolation="linear")

        if log:
            self.to_log.update({
                "percentile_value": percentile_value,
                "percentile_0_5_value": percentile_0_5_value,
                "log_lik_check_coop_std": log_lik_check_coop.std(),
                "log_lik_check_coop_mean": log_lik_check_coop.mean()
            })
        return percentile_value

    def _update_defection_metric(self, epi_defection_metric):
        self.defection_carac_queue.append(epi_defection_metric)
        self.defection_metric = sum(self.defection_carac_queue) / len(self.defection_carac_queue)
        self.to_log["defection_metric"] = round(float(self.defection_metric), 4)


# TODO isn't this already compute in the default policy? (But this may be overwritten in LE)
def compute_s_a_log_likelihood(s, a, algo, no_grad=True, epsilon=1e-12, debug=False):
    """ Get the log_likelihood of the observed action under the algo policy """
    train_batch = SampleBatch({"obs": torch.from_numpy(s).unsqueeze(dim=0)})
    if no_grad:
        with torch.no_grad():
            dist_inputs, _ = algo.model.from_batch(train_batch)
            # Create an action distribution object.
            a_proba_distrib = algo.dist_class(dist_inputs, algo.model)
    else:
        # Pass the training data through our model to get distribution parameters.
        dist_inputs, _ = algo.model.from_batch(train_batch)
        # Create an action distribution object.
        a_proba_distrib = algo.dist_class(dist_inputs, algo.model)

    # Squeeze batch dim since we work on one action only
    assert a_proba_distrib.dist.probs.shape[0] == 1
    observed_actions_all_proba = a_proba_distrib.dist.probs.detach().squeeze(dim=0)

    # Assert action is in discrete space
    assert issubclass(a.dtype.type, numbers.Integral), f"a.dtype: {a.dtype.type}"
    assert a >= 0
    observed_action_idx = a
    observed_action_proba = observed_actions_all_proba[observed_action_idx]
    observed_action_proba = observed_action_proba.clamp(min=epsilon, max=1.0 - epsilon)
    observed_action_log_likelihood = np.array(torch.log(observed_action_proba), dtype=np.float32)

    if debug:
        print("algo.algo_idx", algo.algo_idx)
        print("a_proba_distrib.dist.probs", a_proba_distrib.dist.probs.shape, a_proba_distrib.dist.probs)
        print("observed_action_proba", observed_action_proba)
        print("observed_action_log_likelihood", observed_action_log_likelihood)
    return observed_action_log_likelihood, a_proba_distrib


class LECallBacks(DefaultCallbacks):

    def on_episode_step(self, *, worker, base_env,
                        episode, env_index, **kwargs):
        """Runs on each episode step.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (int): The index of the (vectorized) env, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """
        agent_ids = list(worker.policy_map.keys())
        assert len(agent_ids) == 2, "Implemented for 2 players"
        for agent_id, policy in worker.policy_map.items():
            opp_agent_id = [one_id for one_id in agent_ids if one_id != agent_id][0]
            if hasattr(policy, 'on_episode_step') and callable(policy.on_episode_step):

                # TODO could I move this into the on_postprocess_trajectory callback?
                opp_obs = episode.last_observation_for(opp_agent_id)
                opp_a = episode.last_action_for(opp_agent_id)
                if episode.length > 0:
                    being_punished_by_LE = episode.last_pi_info_for(opp_agent_id).get("punishing", False)
                else:
                    being_punished_by_LE = False

                policy.on_episode_step(opp_obs, opp_a, being_punished_by_LE)

    def on_episode_end(self, *, worker, base_env,
                       policies, episode, env_index, **kwargs):
        """Runs when an episode is done.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (int): The index of the (vectorized) env, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """
        for polidy_ID, policy in policies.items():
            if hasattr(policy, 'on_episode_end') and callable(policy.on_episode_end):
                policy.on_episode_end()
