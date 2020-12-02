from typing import List, Optional, Dict, Tuple
import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID, TensorType
from ray.rllib.policy.policy import Policy
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.postprocessing import discount

from marltoolbox.utils import log

WELFARE_UTILITARIAN = "utilitarian_welfare"
WELFARE_INEQUITY_AVERSION = "inequity_aversion_welfare"
WELFARE_NASH = "nash_welfare"
WELFARE_EGALITARIAN = "egalitarian_welfare"

WELFARES = (WELFARE_UTILITARIAN, WELFARE_INEQUITY_AVERSION)

OPPONENT_ACTIONS = "opponent_actions"
OPPONENT_NEGATIVE_REWARD = "opponent_negative_reward"
DISCOUNTED_RETURNS = "discounted_returns"
REWARDS_UNDER_DEFECTION = "rewards_under_defection"


#
# # TODO make both postprocess_trajectory functions into one
# def postprocess_trajectory(sample_batch: SampleBatch,
#                            other_agent_batches: Optional[Dict[AgentID, Tuple[
#                                "Policy", SampleBatch]]] = None,
#                            episode: Optional["MultiAgentEpisode"] = None,
#                            add_utilitarian_welfare=False,
#                            add_opponent_action=False,
#                            add_opponent_neg_reward=False) -> SampleBatch:
#     """
#     To call in the postprocess_trajectory method of a Policy.
#     Can add the
#     """
#     opp_batches = [batch for policy, batch in list(other_agent_batches.values())]
#
#     if add_utilitarian_welfare:
#         sample_batch = _add_utilitarian_welfare_to_batch(sample_batch, opp_batches)
#     if add_opponent_action:
#         assert len(opp_batches) == 1
#         sample_batch = _add_opponent_action_to_batch(sample_batch, opp_batches[0])
#     if add_opponent_neg_reward:
#         assert len(opp_batches) == 1
#         sample_batch = _add_opponent_neg_reward_to_batch(sample_batch, opp_batches[0])
#
#     return sample_batch
#
# def postprocess_fn(add_utilitarian_welfare=False, add_opponent_action=False, add_opponent_neg_reward=False,
#                    sub_postprocess_fn=None):
#     def postprocess_trajectory(sample_batch: SampleBatch,
#                                other_agent_batches: Optional[Dict[AgentID, Tuple[
#                                    "Policy", SampleBatch]]] = None,
#                                episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
#
#         if sub_postprocess_fn is not None:
#             sample_batch = sub_postprocess_fn(sample_batch, other_agent_batches, episode)
#
#         opp_batches = [batch for policy, batch in list(other_agent_batches.values())]
#
#         if add_utilitarian_welfare:
#             sample_batch = _add_utilitarian_welfare_to_batch(sample_batch, opp_batches)
#         if add_opponent_action:
#             assert len(opp_batches) == 1
#             sample_batch = _add_opponent_action_to_batch(sample_batch, opp_batches[0])
#         if add_opponent_neg_reward:
#             assert len(opp_batches) == 1
#             sample_batch = _add_opponent_neg_reward_to_batch(sample_batch, opp_batches[0])
#         return sample_batch
#     return postprocess_trajectory




class WelfareAndPostprocessCallbacks(log.LoggingCallbacks):

    ADD_UTILITARIAN_WELFARE = False
    ADD_INEQUITY_AVERSION_WELFARE = False
    ADD_EGALITARIAN_WELFARE = False
    ADD_NASH_WELFARE = False

    ADD_OPPONENT_ACTION = False
    ADD_OPPONENT_NEG_REWARD = False

    INEQUITY_AVERSION_ALPHA = 1.0
    INEQUITY_AVERSION_BETA = 0.0
    INEQUITY_AVERSION_GAMMA = 0.9

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, env_index: int, **kwargs):
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

        if self.ADD_EGALITARIAN_WELFARE or self.ADD_NASH_WELFARE:
            # TODO perform rollout in the env
            # Instanciate copy of policies trained with selfish return => can't do that since I would need to create
            # data buffer (to actually train the selfish policy)
            # can't do that here then ?
            # maybe I can:
            # Add a mixin to Policy to :
                # add a nested policy to be trained with selfish rewards + here in on_episode_end add the rollout in
                # extra batch data

            # Here perform rollout + add in batch
            # In a child policy class wrapper for allow access to the selfish alo version
            # Add at each train step => train the selfish too
            raise NotImplementedError()

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
        opp_batches = [batch[1] for batch_policy_id, batch in original_batches.items()
                       if batch_policy_id != policy_id]

        if self.ADD_UTILITARIAN_WELFARE:
            postprocessed_batch = self._add_utilitarian_welfare_to_batch(postprocessed_batch, opp_batches)
        if self.ADD_INEQUITY_AVERSION_WELFARE:
            assert len(opp_batches) == 1
            postprocessed_batch = self._add_inequity_aversion_welfare_to_batch(postprocessed_batch, opp_batches[0],
                                                                          alpha=self.INEQUITY_AVERSION_ALPHA,
                                                                          beta=self.INEQUITY_AVERSION_BETA,
                                                                          gamma=self.INEQUITY_AVERSION_GAMMA)
        if self.ADD_NASH_WELFARE:
            assert len(opp_batches) == 1
            postprocessed_batch = self._add_nash_welfare_to_batch(postprocessed_batch, opp_batches[0])
        if self.ADD_EGALITARIAN_WELFARE:
            assert len(opp_batches) == 1
            postprocessed_batch = self._add_egalitarian_welfare_to_batch(postprocessed_batch, opp_batches[0])
        if self.ADD_OPPONENT_ACTION:
            assert len(opp_batches) == 1
            postprocessed_batch = self._add_opponent_action_to_batch(postprocessed_batch, opp_batches[0])
        if self.ADD_OPPONENT_NEG_REWARD:
            assert len(opp_batches) == 1
            postprocessed_batch = self._add_opponent_neg_reward_to_batch(postprocessed_batch, opp_batches[0])

        super().on_postprocess_trajectory(worker=worker, episode=episode,
            agent_id=agent_id, policy_id=policy_id,
            policies=policies, postprocessed_batch=postprocessed_batch,
            original_batches=original_batches, **kwargs)

    @staticmethod
    def _add_utilitarian_welfare_to_batch(sample_batch: SampleBatch,
                                          opp_ag_batchs: List[SampleBatch]) -> SampleBatch:
        all_batchs_rewards = ([sample_batch[sample_batch.REWARDS]] +
                              [opp_batch[opp_batch.REWARDS] for opp_batch in opp_ag_batchs])
        sample_batch.data[WELFARE_UTILITARIAN] = [sum(reward_points) for reward_points in zip(*all_batchs_rewards)]
        return sample_batch

    @staticmethod
    def _add_opponent_action_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
        sample_batch.data[OPPONENT_ACTIONS] = opp_ag_batch[opp_ag_batch.ACTIONS]
        return sample_batch

    @staticmethod
    def _add_opponent_neg_reward_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
        sample_batch.data[OPPONENT_NEGATIVE_REWARD] = [- opp_r for opp_r in opp_ag_batch[opp_ag_batch.REWARDS]]
        return sample_batch

    @staticmethod
    def _add_inequity_aversion_welfare_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch,
                                                alpha: float, beta: float, gamma: float) -> SampleBatch:
        """
        :param sample_batch: SampleBatch to mutate
        :param opp_ag_batchs:
        :param alpha: coeff of disvalue when own discounted reward is lower than opponent
        :param beta: coeff of disvalue when own discounted reward is lower than opponent
        :param gamma: discount factor
        :return: sample_batch mutated with WELFARE_INEQUITY_AVERSION added
        """

        # TODO verify than this batches are only one full episode
        own_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
        opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
        print("len(own_rewards)", own_rewards.shape, "should equals episode length")
        print("len(opp_rewards)", opp_rewards.shape, "should equals episode length")

        delta = (discount(own_rewards, gamma) - discount(opp_rewards, gamma))
        disvalue_lower_reward = - alpha * delta
        disvalue_higher_reward = beta * delta
        disvalue_lower_reward[disvalue_lower_reward < 0] = 0
        disvalue_higher_reward[disvalue_higher_reward < 0] = 0

        all_batchs_rewards = ([sample_batch[sample_batch.REWARDS]] + disvalue_lower_reward + disvalue_higher_reward)
        sample_batch.data[WELFARE_INEQUITY_AVERSION] = [sum(reward_points) for reward_points in
                                                        zip(*all_batchs_rewards)]
        return sample_batch

    @staticmethod
    def _add_nash_welfare_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:

        # TODO verify than this batches are only one full episode
        own_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
        opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
        own_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])
        opp_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])
        print("len(own_rewards)", own_rewards.shape, "should equals episode length")
        print("len(opp_rewards)", opp_rewards.shape, "should equals episode length")

        own_delta = (sum(own_rewards) - sum(own_rewards_under_defection))
        opp_delta = (sum(opp_rewards) - sum(opp_rewards_under_defection))

        nash_welfare = own_delta * opp_delta

        sample_batch.data[WELFARE_NASH] = ([0.0] * (len(sample_batch[sample_batch.REWARDS]) - 1)) + [nash_welfare]
        return sample_batch

    @staticmethod
    def _add_egalitarian_welfare_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:

        # TODO verify than this batches are only one full episode
        own_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
        opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
        own_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])
        opp_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])
        print("len(own_rewards)", own_rewards.shape, "should equals episode length")
        print("len(opp_rewards)", opp_rewards.shape, "should equals episode length")

        own_delta = (sum(own_rewards) - sum(own_rewards_under_defection))
        opp_delta = (sum(opp_rewards) - sum(opp_rewards_under_defection))

        egalitarian_welfare = min(own_delta, opp_delta)

        sample_batch.data[WELFARE_EGALITARIAN] = ([0.0] * (len(sample_batch[sample_batch.REWARDS]) - 1)) + [egalitarian_welfare]
        return sample_batch
