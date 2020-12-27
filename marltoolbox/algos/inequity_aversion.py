##########
# Code from: https://github.com/julianstastny/implicit-bargaining-problems/blob/master/inequity_aversion.py
##########

import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG
from ray.rllib.utils import merge_dicts

# TODO this is not working since the advantage is copute before modifying the reward with inequity aversion?
# TODO Use what is in postprocessing
def reward_postprocessing(policy, rollout, other_agent_batches, episode):
    completed = rollout[SampleBatch.DONES][-1]

    # Add advantages (like in the original A3C)
    if completed:
        last_r = 0.0
    else:
        last_r = policy._value(rollout[SampleBatch.NEXT_OBS][-1])

    rollout = compute_advantages(
        rollout, last_r, policy.config["gamma"], policy.config["lambda"],
        policy.config["use_gae"], policy.config["use_critic"])

    # Compute inequility aversion
    self_reward = rollout[SampleBatch.REWARDS]
    [(_, opponent_rollout)] = list(other_agent_batches.values())
    other_reward = opponent_rollout[SampleBatch.REWARDS]
    # other_reward = other_agent_batches["player_2"][SampleBatch.REWARDS]
    smooth_self_reward = np.empty_like(self_reward)
    smooth_other_reward = np.empty_like(other_reward)

    prev_self = 0
    prev_other = 0
    for i, (s_r, o_r) in enumerate(zip(self_reward, other_reward)):
        smooth_self_reward[i] = policy.config["inequity_lambda"] * policy.config["gamma"] * prev_self + self_reward[i]
        smooth_other_reward[i] = policy.config["inequity_lambda"] * policy.config["gamma"] * prev_other + other_reward[
            i]
        prev_self = smooth_self_reward[i]
        prev_other = smooth_other_reward[i]
    # smooth_self_reward, smooth_other_reward = policy.update_and_get_smooth_rewards(self_reward, other_reward)
    disadvantage = np.clip(smooth_other_reward - smooth_self_reward, 0, None)
    advantage = np.clip(smooth_self_reward - smooth_other_reward, 0, None)
    utility = self_reward - policy.config["inequity_alpha"] * disadvantage - policy.config["inequity_beta"] * advantage
    rollout[SampleBatch.REWARDS] = utility
    return rollout


InequityAverseTorchPolicy = A3CTorchPolicy.with_updates(
    name="InequityAverseA3C",
    postprocess_fn=reward_postprocessing,
)

InequityAversionTrainer = A3CTrainer.with_updates(
    default_policy=InequityAverseTorchPolicy,
    get_policy_class=None,
    default_config=merge_dicts(
        A3C_CONFIG,
        {
            "inequity_alpha": 0,
            "inequity_beta": 1,
            "inequity_lambda": 0.5
        }
    )
)