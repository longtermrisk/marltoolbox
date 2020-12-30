import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import discount
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import List, Dict

WELFARE_UTILITARIAN = "utilitarian_welfare"
WELFARE_INEQUITY_AVERSION = "inequity_aversion_welfare"
WELFARE_NASH = "nash_welfare"
WELFARE_EGALITARIAN = "egalitarian_welfare"

WELFARES = (WELFARE_UTILITARIAN, WELFARE_INEQUITY_AVERSION)

OPPONENT_ACTIONS = "opponent_actions"
OPPONENT_NEGATIVE_REWARD = "opponent_negative_reward"
DISCOUNTED_RETURNS = "discounted_returns"
REWARDS_UNDER_DEFECTION = "rewards_under_defection"


def get_postprocessing_welfare_function(
        add_utilitarian_welfare=False,
        add_egalitarian_welfare=False, add_nash_welfare=False,
        add_opponent_action=False, add_opponent_neg_reward=False,
        add_inequity_aversion_welfare=False,
        inequity_aversion_alpha=0.0, inequity_aversion_beta=1.0,
        inequity_aversion_gamma=0.9, inequity_aversion_lambda=0.9,
        additional_fn: list = []):
    def postprocess_fn(policy, sample_batch, other_agent_batches, episode):

        if add_utilitarian_welfare:
            opp_batches = [v[1] for v in other_agent_batches.values()]
            sample_batch = _add_utilitarian_welfare_to_batch(sample_batch, opp_batches)
        if add_inequity_aversion_welfare:
            assert len(other_agent_batches) == 1
            opp_batch = other_agent_batches[list(other_agent_batches.keys())[0]][1]
            sample_batch = _add_inequity_aversion_welfare_to_batch(
                sample_batch, opp_batch,
                alpha=inequity_aversion_alpha,
                beta=inequity_aversion_beta,
                gamma=inequity_aversion_gamma,
                lambda_=inequity_aversion_lambda)
        if add_nash_welfare:
            assert len(other_agent_batches) == 1
            opp_batch = other_agent_batches[list(other_agent_batches.keys())[0]][1]
            sample_batch = _add_nash_welfare_to_batch(sample_batch, opp_batch)
        if add_egalitarian_welfare:
            assert len(other_agent_batches) == 1
            opp_batch = other_agent_batches[list(other_agent_batches.keys())[0]][1]
            sample_batch = _add_egalitarian_welfare_to_batch(sample_batch, opp_batch)
        if add_opponent_action:
            assert len(other_agent_batches) == 1
            opp_batch = other_agent_batches[list(other_agent_batches.keys())[0]][1]
            sample_batch = _add_opponent_action_to_batch(sample_batch, opp_batch)
        if add_opponent_neg_reward:
            assert len(other_agent_batches) == 1
            opp_batch = other_agent_batches[list(other_agent_batches.keys())[0]][1]
            sample_batch = _add_opponent_neg_reward_to_batch(sample_batch, opp_batch)

        for postprocessing_function in additional_fn:
            sample_batch = postprocessing_function(policy, sample_batch, other_agent_batches, episode)

        return sample_batch

    return postprocess_fn


def _add_utilitarian_welfare_to_batch(sample_batch: SampleBatch,
                                      opp_ag_batchs: List[SampleBatch]) -> SampleBatch:
    all_batchs_rewards = ([sample_batch[sample_batch.REWARDS]] +
                          [opp_batch[opp_batch.REWARDS] for opp_batch in opp_ag_batchs])
    sample_batch.data[WELFARE_UTILITARIAN] = np.array(
        [sum(reward_points) for reward_points in zip(*all_batchs_rewards)])
    return sample_batch


def _add_opponent_action_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
    sample_batch.data[OPPONENT_ACTIONS] = opp_ag_batch[opp_ag_batch.ACTIONS]
    return sample_batch


def _add_opponent_neg_reward_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
    sample_batch.data[OPPONENT_NEGATIVE_REWARD] = np.array([- opp_r for opp_r in opp_ag_batch[opp_ag_batch.REWARDS]])
    return sample_batch


# TODO maybe should not apply gamma (only lambda)
def _add_inequity_aversion_welfare_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch,
                                            alpha: float, beta: float, gamma: float,
                                            lambda_: float) -> SampleBatch:
    """
    :param sample_batch: SampleBatch to mutate
    :param opp_ag_batchs:
    :param alpha: coeff of disvalue when own discounted reward is lower than opponent
    :param beta: coeff of disvalue when own discounted reward is hihger than opponent
    :param gamma: discount factor
    :return: sample_batch mutated with WELFARE_INEQUITY_AVERSION added
    """

    # TODO verify than this batches are only one full episode
    own_rewards = np.array(sample_batch[sample_batch.REWARDS])
    opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    own_rewards = np.flip(own_rewards)
    opp_rewards = np.flip(opp_rewards)
    delta = (discount(own_rewards, gamma * lambda_) - discount(opp_rewards, gamma * lambda_))
    delta = np.flip(delta)
    disvalue_lower_than_opp = alpha * (-delta)
    disvalue_higher_than_opp = beta * delta
    disvalue_lower_than_opp[disvalue_lower_than_opp < 0] = 0
    disvalue_higher_than_opp[disvalue_higher_than_opp < 0] = 0

    welfare = sample_batch[sample_batch.REWARDS] - disvalue_lower_than_opp - disvalue_higher_than_opp

    sample_batch.data[WELFARE_INEQUITY_AVERSION] = welfare

    # print("inequity aversion welfare", welfare)
    return sample_batch


def _add_nash_welfare_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
    # TODO verify than this batches are only one full episode
    own_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    own_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])
    opp_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])

    own_delta = (sum(own_rewards) - sum(own_rewards_under_defection))
    opp_delta = (sum(opp_rewards) - sum(opp_rewards_under_defection))

    nash_welfare = own_delta * opp_delta

    sample_batch.data[WELFARE_NASH] = ([0.0] * (len(sample_batch[sample_batch.REWARDS]) - 1)) + [nash_welfare]
    return sample_batch


def _add_egalitarian_welfare_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
    # TODO verify than this batches are only one full episode
    own_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    own_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])
    opp_rewards_under_defection = np.array(opp_ag_batch.data[REWARDS_UNDER_DEFECTION])

    own_delta = (sum(own_rewards) - sum(own_rewards_under_defection))
    opp_delta = (sum(opp_rewards) - sum(opp_rewards_under_defection))

    egalitarian_welfare = min(own_delta, opp_delta)

    sample_batch.data[WELFARE_EGALITARIAN] = ([0.0] * (len(sample_batch[sample_batch.REWARDS]) - 1)) + [
        egalitarian_welfare]
    return sample_batch


class OverwriteRewardWtWelfareCallback(DefaultCallbacks):

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs):

        assert sum([k in WELFARES for k in postprocessed_batch.data.keys()]) <= 1, "only one welfare must be available"

        for welfare_key in WELFARES:
            if welfare_key in postprocessed_batch.data.keys():
                postprocessed_batch[postprocessed_batch.REWARDS] = postprocessed_batch.data[welfare_key]
                break

        return postprocessed_batch
