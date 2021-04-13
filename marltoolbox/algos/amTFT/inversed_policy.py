from ray.rllib.utils.annotations import override

from marltoolbox.algos.amTFT import base_policy, policy_using_rollouts


class InversedAmTFTRolloutsTorchPolicy(
    policy_using_rollouts.AmTFTRolloutsTorchPolicy):
    """
    Instead of simulating the opponent, simulate our own policy and act as
    if it was the opponent.
    """

    def _init_for_rollout(self, config):
        super()._init_for_rollout(config)
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

    # @override(policy_using_rollouts.AmTFTRolloutsTorchPolicy)
    # def _select_algo_to_use_in_eval(self):
    #     assert self.performing_rollouts
    #
    #     if not self.use_opponent_policies:
    #         if self.n_steps_to_punish == 0:
    #             self.active_algo_idx = base.OPP_COOP_POLICY_IDX
    #         elif self.n_steps_to_punish > 0:
    #             self.active_algo_idx = base.OPP_SELFISH_POLICY_IDX
    #             self.n_steps_to_punish -= 1
    #         else:
    #             raise ValueError("self.n_steps_to_punish can't be below zero")
    #     else:
    #         # assert self.performing_rollouts
    #         if self.n_steps_to_punish_opponent == 0:
    #             self.active_algo_idx = base.OWN_COOP_POLICY_IDX
    #         elif self.n_steps_to_punish_opponent > 0:
    #             self.active_algo_idx = base.OWN_SELFISH_POLICY_IDX
    #             self.n_steps_to_punish_opponent -= 1
    #         else:
    #             raise ValueError("self.n_steps_to_punish_opp "
    #                              "can't be below zero")

    # @override(policy_using_rollouts.AmTFTRolloutsTorchPolicy)
    # def _init_for_rollout(self, config):
    #     super()._init_for_rollout(config)
    #     # the policies stored as opponent_policies are our own policy
    #     # (not the opponent's policies)
    #     self.use_opponent_policies = False

    # @override(policy_using_rollouts.AmTFTRolloutsTorchPolicy)
    # def _prepare_to_perform_virtual_rollouts_in_env(self, worker):
    #     outputs = super()._prepare_to_perform_virtual_rollouts_in_env(
    #         worker)
    #     # the policies stored as opponent_policies are our own policy
    #     # (not the opponent's policies)
    #     self.use_opponent_policies = True
    #     return outputs

    # @override(policy_using_rollouts.AmTFTRolloutsTorchPolicy)
    # def _stop_performing_virtual_rollouts_in_env(self, n_steps_to_punish):
    #     super()._stop_performing_virtual_rollouts_in_env(n_steps_to_punish)
    #     # the policies stored as opponent_policies are our own policy
    #     # (not the opponent's policies)
    #     self.use_opponent_policies = True

    # @override(policy_using_rollouts.AmTFTRolloutsTorchPolicy)
    # def compute_actions(
    #         self,
    #         obs_batch: Union[List[TensorType], TensorType],
    #         state_batches: Optional[List[TensorType]] = None,
    #         prev_action_batch: Union[List[TensorType], TensorType] = None,
    #         prev_reward_batch: Union[List[TensorType], TensorType] = None,
    #         info_batch: Optional[Dict[str, list]] = None,
    #         episodes: Optional[List["MultiAgentEpisode"]] = None,
    #         explore: Optional[bool] = None,
    #         timestep: Optional[int] = None,
    #         **kwargs) -> \
    #         Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
    #
    #     # Option to overwrite action during internal rollouts
    #     if self.use_opponent_policies:
    #         if len(self.overwrite_action) > 0:
    #             actions, state_out, extra_fetches = \
    #                 self.overwrite_action.pop(0)
    #             if self.verbose > 1:
    #                 print("overwrite actions", actions, type(actions))
    #             return actions, state_out, extra_fetches
    #
    #     return super().compute_actions(
    #         obs_batch, state_batches, prev_action_batch, prev_reward_batch,
    #         info_batch, episodes, explore, timestep, **kwargs)

# debit = self._compute_debit(
#     last_obs, opp_action, worker, base_env,
#     episode, env_index, coop_opp_simulated_action)

# self.n_steps_to_punish = self._compute_punishment_duration(
#     opp_action,
#     coop_opp_simulated_action,
#     worker,
#     last_obs)
