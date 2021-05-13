from ray.rllib.utils.annotations import override

from marltoolbox.algos.amTFT import base_policy, policy_using_rollouts


class InversedAmTFTRolloutsTorchPolicy(
    policy_using_rollouts.AmTFTRolloutsTorchPolicy
):
    """
    Instead of simulating the opponent, simulate our own policy and act as
    if it was the opponent.
    """

    def _init(self, config):
        super()._init(config)
        self.ag_id_rollout_reward_to_read = self.own_policy_id

    @override(base_policy.AmTFTPolicyBase)
    def _get_information_from_opponent(self, agent_id, agent_ids, episode):
        own_agent_id = agent_id
        own_a = episode.last_action_for(own_agent_id)

        return self.own_previous_obs, self.both_previous_raw_obs, own_a

    @override(policy_using_rollouts.AmTFTRolloutsTorchPolicy)
    def _switch_own_and_opp(self, agent_id):
        output = super()._switch_own_and_opp(agent_id)
        self.use_opponent_policies = not self.use_opponent_policies
        return output
