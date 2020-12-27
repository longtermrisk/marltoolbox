##########
# Code modified from: https://github.com/Manuscrit/SLM-Lab/blob/support_multi_agents_with_awareness/slm_lab/agent/algorithm/meta_algorithm/learning_equilibrium.py
##########

import copy
from collections import deque
import logging

import numpy as np
import torch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.utils.typing import TensorType
from typing import List, Union, Optional, Dict, Tuple

from marltoolbox.algos import hierarchical
from marltoolbox.utils import postprocessing

torch, nn = try_import_torch()

LE_DEFAULT_CONFIG_UPDATE = merge_dicts(
    hierarchical.DEFAULT_CONFIG,
    {
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
    }
)


# TODO make a parent class for nested/hierarchical algo?
class LE(hierarchical.HierarchicalTorchPolicy):
    """
    Learning Tit-for-tat (Learning Equilibrium(LE))
    """
    EPSILON = 1e-12
    DEBUG = False

    COOP_POLICY_IDX = 0
    PUNITIVE_POLICY_IDX = 1
    COOP_OPP_POLICY_IDX = 2
    SPL_OPP_POLICY_IDX = 3

    WARMUP_LENGTH = 0
    INITIALLY_ACTIVE_ALGO = COOP_POLICY_IDX

    def __init__(self, observation_space, action_space, config, **kwargs):

        super().__init__(observation_space, action_space, config,
                         after_init_nested=(lambda policy: [torch.nn.init.normal_(p, mean=0.0, std=0.1)
                                                            for p in policy.model.parameters()]),
                         **kwargs)

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
        log_len_in_steps = 100
        # for 0) the cooperative policy
        self.action_pd_coop = deque(maxlen=log_len_in_steps)
        # for 1) the punitive policy
        self.action_pd_punish = deque(maxlen=log_len_in_steps)
        # for 2) the estimated opponent cooperative policy
        self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
        # for 3) the estimated opponent policy
        self.action_pd_opp = deque(maxlen=log_len_in_steps)
        self.n_cooperation_steps_in_current_epi = 0
        self.n_punishment_steps_in_current_epi = 0

        self.add_welfare_fn = postprocessing.get_postprocessing_welfare_function(
            add_utilitarian_welfare=True, add_opponent_action=True, add_opponent_neg_reward=True)

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
        assert len(obs_batch) == 1, "LE only works with sampling one step at a time"

        actions, state_out, extra_fetches = super().compute_actions(obs_batch, state_batches, prev_action_batch,
                                                                    prev_reward_batch, info_batch, episodes, explore,
                                                                    timestep, **kwargs)

        extra_fetches['punishing'] = [self.active_algo_idx == self.PUNITIVE_POLICY_IDX for _ in obs_batch]

        return actions, [], extra_fetches

    def learn_on_batch(self, samples: SampleBatch):
        learner_stats = {"learner_stats": {}}

        # Update LR used in optimizer
        self.optimizer()

        for i, algo in enumerate(self.algorithms):

            samples_copy = samples.copy()
            if i == self.COOP_POLICY_IDX or i == self.COOP_OPP_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[
                                                                  postprocessing.WELFARE_UTILITARIAN])
                samples_copy = self._filter_sample_batch(samples_copy, remove=True, filter_key="punishing")
            elif i == self.PUNITIVE_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[
                                                                  postprocessing.OPPONENT_NEGATIVE_REWARD])
            elif i == self.COOP_OPP_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[
                                                                  postprocessing.WELFARE_UTILITARIAN])
                samples_copy = self._filter_sample_batch(samples_copy, remove=True, filter_key="being_punished")
            elif i == self.SPL_OPP_POLICY_IDX:
                samples_copy[samples_copy.ACTIONS] = np.array(samples_copy.data[
                                                                  postprocessing.OPPONENT_ACTIONS])
                samples_copy = self._filter_sample_batch(samples_copy, remove=True, filter_key="being_punished")

            # TODO currently continue to learn because I use the batch to stop but the batch is sampled!
            if len(samples_copy[samples_copy.ACTIONS]) > 0:
                learner_stats["learner_stats"][f"learner_stats_algo{i}"] = algo.learn_on_batch(samples_copy)
                self.to_log[f'algo{i}_cur_lr'] = algo.cur_lr
            # For debugging purpose log the true lr (to be compared to algo.cur_lr)
            for j, opt in enumerate(algo._optimizers):
                self.to_log[f"algo_{i}_{j}_lr"] = [p["lr"] for p in opt.param_groups][0]

        return learner_stats

    def on_episode_step(self, opp_obs, opp_a, being_punished_by_LE):

        self.being_punished_by_LE = being_punished_by_LE

        if not self.being_punished_by_LE:
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
            print("DEFECTION DETECTED")

        if self.remaining_punishing_time <= 0:
            self.active_algo_idx = self.COOP_POLICY_IDX

        self.to_log["remaining_punishing_time"] = self.remaining_punishing_time
        self.to_log["active_algo_idx"] = self.active_algo_idx
        self.to_log["detected_defection"] = self.detected_defection
        self.to_log["being_punished_by_LE"] = self.being_punished_by_LE
        self.detected_defection = False

    def postprocess_trajectory(
            self, sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[
                "Policy", SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        return self.add_welfare_fn(self, sample_batch, other_agent_batches, episode)

    def _put_log_likelihood_in_data_buffer(self, s, a, data_queue, log=True):
        s = torch.from_numpy(s).unsqueeze(dim=0)
        log_likelihood_opponent_cooperating = compute_log_likelihoods_wt_exploration(
            self.algorithms[self.COOP_OPP_POLICY_IDX], a, s)
        log_likelihood_approximated_opponent = compute_log_likelihoods_wt_exploration(
            self.algorithms[self.SPL_OPP_POLICY_IDX], a, s)
        log_likelihood_opponent_cooperating = float(log_likelihood_opponent_cooperating)
        log_likelihood_approximated_opponent = float(log_likelihood_approximated_opponent)

        self.to_log["log_likelihood_opponent_cooperating"] = log_likelihood_opponent_cooperating
        self.to_log["log_likelihood_approximated_opponent"] = log_likelihood_approximated_opponent
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


# Modified from torch_policy_template
def compute_log_likelihoods_wt_exploration(
        policy,
        actions: Union[List[TensorType], TensorType],
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[Union[List[TensorType],
                                          TensorType]] = None,
        prev_reward_batch: Optional[Union[List[
                                              TensorType], TensorType]] = None) -> TensorType:
    if policy.action_sampler_fn and policy.action_distribution_fn is None:
        raise ValueError("Cannot compute log-prob/likelihood w/o an "
                         "`action_distribution_fn` and a provided "
                         "`action_sampler_fn`!")

    with torch.no_grad():
        input_dict = policy._lazy_tensor_dict({
            SampleBatch.CUR_OBS: obs_batch,
            SampleBatch.ACTIONS: actions
        })
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)

        # Exploration hook before each forward pass.
        policy.exploration.before_compute_actions(explore=False)

        # Action dist class and inputs are generated via custom function.
        if policy.action_distribution_fn:
            dist_inputs, dist_class, _ = policy.action_distribution_fn(
                policy=policy,
                model=policy.model,
                obs_batch=input_dict[SampleBatch.CUR_OBS],
                explore=False,
                is_training=False)
        # Default action-dist inputs calculation.
        else:
            dist_class = policy.dist_class
            dist_inputs, _ = policy.model(input_dict, state_batches,
                                          seq_lens)

        action_dist = dist_class(dist_inputs, policy.model)
        if policy.config["explore"]:
            # TODO Adding that because of a bug in TorchCategorical which modify dist_inputs through action_dist:
            _, _ = policy.exploration.get_exploration_action(
                action_distribution=action_dist,
                timestep=policy.global_timestep,
                explore=policy.config["explore"])
            action_dist = dist_class(dist_inputs, policy.model)

        log_likelihoods = action_dist.logp(input_dict[SampleBatch.ACTIONS])
        return log_likelihoods


class LECallbacks(DefaultCallbacks):

    def on_episode_step(self, *, worker, base_env,
                        episode, env_index, **kwargs):
        agent_ids = list(worker.policy_map.keys())
        assert len(agent_ids) == 2, "Implemented for 2 players"
        for agent_id, policy in worker.policy_map.items():
            opp_agent_id = [one_id for one_id in agent_ids if one_id != agent_id][0]
            if hasattr(policy, 'on_episode_step') and callable(policy.on_episode_step):
                opp_obs = episode.last_observation_for(opp_agent_id)
                opp_a = episode.last_action_for(opp_agent_id)
                if episode.length > 0:
                    being_punished_by_LE = episode.last_pi_info_for(opp_agent_id).get("punishing", False)
                    policy.on_episode_step(opp_obs, opp_a, being_punished_by_LE)
                else:
                    # Only during init
                    policy.on_episode_step(opp_obs, opp_a, False)

    def on_episode_end(self, *, worker, base_env,
                       policies, episode, env_index, **kwargs):
        for polidy_ID, policy in policies.items():
            if hasattr(policy, 'on_episode_end') and callable(policy.on_episode_end):
                policy.on_episode_end()

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs):

        all_agent_keys = list(original_batches.keys())
        all_agent_keys.remove(agent_id)
        assert len(all_agent_keys) == 1, "LE only works for 2 agents"
        opp_policy_id = all_agent_keys[0]

        opp_is_broadcast_punishment_state = "punishing" in original_batches[opp_policy_id][1].data.keys()
        if opp_is_broadcast_punishment_state:
            postprocessed_batch.data['being_punished'] = \
                copy.deepcopy(original_batches[all_agent_keys[0]][1].data["punishing"])
        else:
            postprocessed_batch.data['being_punished'] = \
                [False] * len(postprocessed_batch[postprocessed_batch.OBS])
