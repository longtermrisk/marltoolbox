##########
# Code modified from: https://github.com/Manuscrit/SLM-Lab/blob/support_multi_agents_with_awareness/slm_lab/agent/algorithm/meta_algorithm/learning_equilibrium.py
# WORK IN PROGRESS!! This implementation is not working properly.
##########

import copy
from collections import deque
from typing import List, Union, Optional, Dict, Tuple

import numbers
import numpy as np

import torch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID, TensorType
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import (FLOAT_MIN, huber_loss,
                                       reduce_mean_ignore_inf,
                                       softmax_cross_entropy_with_logits)
from ray.rllib.utils import merge_dicts

from marltoolbox.utils import preprocessing, log
from marltoolbox.algos import hierarchical

torch, nn = try_import_torch()

LE_DEFAULT_CONFIG_UPDATE = merge_dicts(
    hierarchical.HIERARCHICAL_DEFAULT_CONFIG_UPDATE,
    {
        # LE hyper-parameters
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
    Learning Tit-for-tat or Learning Equilibrium(LE)
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

            # Log true lr
            # TODO Here it is only logging for the 1st parameter
            for j, opt in enumerate(algo._optimizers):
                self.to_log[f"algo_{i}_{j}_lr"] = [p["lr"] for p in opt.param_groups][0]

            # TODO use the RLLib batch format directly
            samples_copy = samples.copy()
            if i == self.COOP_POLICY_IDX or i == self.COOP_OPP_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[
                                                                  preprocessing.WELFARE_UTILITARIAN])
                samples_copy = self._filter_sample_batch(samples_copy, remove=True, filter_key="punishing")
            elif i == self.PUNITIVE_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[
                                                                  preprocessing.OPPONENT_NEGATIVE_REWARD])
            elif i == self.COOP_OPP_POLICY_IDX:
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[
                                                                  preprocessing.WELFARE_UTILITARIAN])
                samples_copy = self._filter_sample_batch(samples_copy, remove=True, filter_key="being_punished")
            elif i == self.SPL_OPP_POLICY_IDX:
                samples_copy[samples_copy.ACTIONS] = np.array(samples_copy.data[
                                                                  preprocessing.OPPONENT_ACTIONS])
                samples_copy = self._filter_sample_batch(samples_copy, remove=True, filter_key="being_punished")

            # TODO currently continue to learn because I use the batch to stop but the batch is sampled!
            # print("len(samples_copy[samples_copy.ACTIONS])", len(samples_copy[samples_copy.ACTIONS]))
            if len(samples_copy[samples_copy.ACTIONS]) > 0:
                learner_stats["learner_stats"][f"learner_stats_algo{i}"] = algo.learn_on_batch(samples_copy)
                self.to_log[f'algo{i}_cur_lr'] = algo.cur_lr

        return learner_stats

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

    # def postprocess_trajectory(
    #         self,
    #         sample_batch: SampleBatch,
    #         other_agent_batches: Optional[Dict[AgentID, Tuple[
    #             "Policy", SampleBatch]]] = None,
    #         episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
    #     """Implements algorithm-specific trajectory postprocessing.
    #
    #     This will be called on each trajectory fragment computed during policy
    #     evaluation. Each fragment is guaranteed to be only from one episode.
    #
    #     Args:
    #         sample_batch (SampleBatch): batch of experiences for the policy,
    #             which will contain at most one episode trajectory.
    #         other_agent_batches (dict): In a multi-agent env, this contains a
    #             mapping of agent ids to (policy, agent_batch) tuples
    #             containing the policy and experiences of the other agents.
    #         episode (Optional[MultiAgentEpisode]): An optional multi-agent
    #             episode object to provide access to all of the
    #             internal episode state, which may be useful for model-based or
    #             multi-agent algorithms.
    #
    #     Returns:
    #         SampleBatch: Postprocessed sample batch.
    #     """
    #     return preprocessing.postprocess_trajectory(sample_batch, other_agent_batches, episode,
    #                                                       add_utilitarian_welfare=True,
    #                                                       add_opponent_action=True,
    #                                                       add_opponent_neg_reward=True)

    def _put_log_likelihood_in_data_buffer(self, s, a, data_queue, log=True):
        s = torch.from_numpy(s).unsqueeze(dim=0)
        log_likelihood_opponent_cooperating = compute_log_likelihoods_wt_exploration(self.algorithms[self.COOP_OPP_POLICY_IDX], a, s)
        log_likelihood_approximated_opponent = compute_log_likelihoods_wt_exploration(self.algorithms[self.SPL_OPP_POLICY_IDX], a, s)
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


class LECallBacks(preprocessing.WelfareAndPostprocessCallbacks):

    ADD_UTILITARIAN_WELFARE = True
    ADD_OPPONENT_ACTION = True
    ADD_OPPONENT_NEG_REWARD = True
    ADD_INEQUITY_AVERSION_WELFARE = False

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
                #  on_postprocess_trajectory seems to be call in the local_worker (used for training) and just after
                #  a sampling by a rollout worker => can't use it
                opp_obs = episode.last_observation_for(opp_agent_id)
                opp_a = episode.last_action_for(opp_agent_id)
                if episode.length > 0:
                    being_punished_by_LE = episode.last_pi_info_for(opp_agent_id).get("punishing", False)
                    policy.on_episode_step(opp_obs, opp_a, being_punished_by_LE)
                else:
                    # Only during init
                    policy.on_episode_step(opp_obs, opp_a, False)

        super().on_episode_step(worker=worker, base_env=base_env,
                        episode=episode, env_index=env_index, **kwargs)

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

        super().on_episode_end(worker=worker, base_env=base_env, policies=policies,
                        episode=episode, env_index=env_index, **kwargs)

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs):
        """Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            episode (MultiAgentEpisode): Episode object.
            agent_id (str): Id of the current agent.
            policy_id (str): Id of the current policy for the agent.
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            postprocessed_batch (SampleBatch): The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches (dict): Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """

        all_agent_keys = list(original_batches.keys())
        all_agent_keys.remove(agent_id)
        assert len(all_agent_keys) == 1

        opponent_broadcast_punish_state = "punishing" in original_batches[all_agent_keys[0]][1].data.keys()
        if opponent_broadcast_punish_state:
            postprocessed_batch.data['being_punished'] = copy.deepcopy(
                original_batches[all_agent_keys[0]][1].data["punishing"])
        else:
            postprocessed_batch.data['being_punished'] = [False] * len(postprocessed_batch[postprocessed_batch.OBS])

        super().on_postprocess_trajectory(worker=worker, episode=episode,
            agent_id=agent_id, policy_id=policy_id,
            policies=policies, postprocessed_batch=postprocessed_batch,
            original_batches=original_batches, **kwargs)


