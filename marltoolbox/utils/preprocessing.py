from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID
from typing import List, Optional, Dict, Tuple

UTILITARIAN_WELFARE_KEY = "utilitarian_welfare"
OPPONENT_ACTIONS_KEY = "opponent_actions"
OPPONENT_NEGATIVE_REWARD_KEY = "opponent_negative_reward"

def _add_utilitarian_welfare_to_batch(sample_batch: SampleBatch, opp_ag_batchs: List[SampleBatch]) -> SampleBatch:
    all_batchs_rewards = ([sample_batch[sample_batch.REWARDS]] +
                          [opp_batch[opp_batch.REWARDS] for opp_batch in opp_ag_batchs])
    sample_batch.data[UTILITARIAN_WELFARE_KEY] = [sum(reward_points) for reward_points in zip(*all_batchs_rewards)]
    return sample_batch


def _add_opponent_action_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
    sample_batch.data[OPPONENT_ACTIONS_KEY] = opp_ag_batch[opp_ag_batch.ACTIONS]
    return sample_batch

def _add_opponent_neg_reward_to_batch(sample_batch: SampleBatch, opp_ag_batch: SampleBatch) -> SampleBatch:
    sample_batch.data[OPPONENT_NEGATIVE_REWARD_KEY] = [- opp_r for opp_r in opp_ag_batch[opp_ag_batch.REWARDS]]
    return sample_batch


# TODO make both postprocess_trajectory functions into one
def postprocess_trajectory(sample_batch: SampleBatch,
                           other_agent_batches: Optional[Dict[AgentID, Tuple[
                               "Policy", SampleBatch]]] = None,
                           episode: Optional["MultiAgentEpisode"] = None,
                           add_utilitarian_welfare=False,
                           add_opponent_action=False,
                           add_opponent_neg_reward=False) -> SampleBatch:
    """
    To call in the postprocess_trajectory method of a Policy.
    Can add the
    """
    opp_batches = [batch for policy, batch in list(other_agent_batches.values())]

    if add_utilitarian_welfare:
        sample_batch = _add_utilitarian_welfare_to_batch(sample_batch, opp_batches)
    if add_opponent_action:
        assert len(opp_batches) == 1
        sample_batch = _add_opponent_action_to_batch(sample_batch, opp_batches[0])
    if add_opponent_neg_reward:
        assert len(opp_batches) == 1
        sample_batch = _add_opponent_neg_reward_to_batch(sample_batch, opp_batches[0])

    return sample_batch

def postprocess_fn(add_utilitarian_welfare=False, add_opponent_action=False, add_opponent_neg_reward=False,
                   sub_postprocess_fn=None):
    def postprocess_trajectory(sample_batch: SampleBatch,
                               other_agent_batches: Optional[Dict[AgentID, Tuple[
                                   "Policy", SampleBatch]]] = None,
                               episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:

        if sub_postprocess_fn is not None:
            sample_batch = sub_postprocess_fn(sample_batch, other_agent_batches, episode)

        opp_batches = [batch for policy, batch in list(other_agent_batches.values())]

        if add_utilitarian_welfare:
            sample_batch = _add_utilitarian_welfare_to_batch(sample_batch, opp_batches)
        if add_opponent_action:
            assert len(opp_batches) == 1
            sample_batch = _add_opponent_action_to_batch(sample_batch, opp_batches[0])
        if add_opponent_neg_reward:
            assert len(opp_batches) == 1
            sample_batch = _add_opponent_neg_reward_to_batch(sample_batch, opp_batches[0])
        return sample_batch
    return postprocess_trajectory