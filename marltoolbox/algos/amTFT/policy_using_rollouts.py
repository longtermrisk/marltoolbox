from typing import List, Union, Optional, Dict, Tuple

import numpy as np
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import TensorType

from marltoolbox.algos.amTFT.base import (
    OWN_COOP_POLICY_IDX,
    OWN_SELFISH_POLICY_IDX,
    OPP_SELFISH_POLICY_IDX,
    OPP_COOP_POLICY_IDX,
    WORKING_STATES,
)
from marltoolbox.algos.amTFT.base_policy import AmTFTPolicyBase
from marltoolbox.utils import rollout


class AmTFTRolloutsTorchPolicy(AmTFTPolicyBase):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)
        self._init_for_rollout(self.config)

    def _init_for_rollout(self, config):
        self.last_k = config["last_k"]
        self.use_opponent_policies = False
        self.rollout_length = config["rollout_length"]
        self.n_rollout_replicas = config["n_rollout_replicas"]
        self.performing_rollouts = False
        self.overwrite_action = []
        self.own_policy_id = config["own_policy_id"]
        self.opp_policy_id = config["opp_policy_id"]
        self.n_steps_to_punish_opponent = 0
        self.ag_id_rollout_reward_to_read = self.opp_policy_id

        # Don't support LSTM (at least because of action
        # overwriting needed in the rollouts)
        if "model" in config.keys():
            if "use_lstm" in config["model"].keys():
                assert not config["model"]["use_lstm"]

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
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        # Option to overwrite action during internal rollouts
        if self.use_opponent_policies:
            if len(self.overwrite_action) > 0:
                actions, state_out, extra_fetches = self.overwrite_action.pop(
                    0
                )
                if self.verbose > 1:
                    print("overwritten actions", actions, type(actions))
                return actions, state_out, extra_fetches

        return super().compute_actions(
            obs_batch,
            state_batches,
            prev_action_batch,
            prev_reward_batch,
            info_batch,
            episodes,
            explore,
            timestep,
            **kwargs,
        )

    def _select_algo_to_use_in_eval(self):
        if not self.use_opponent_policies:
            if self.n_steps_to_punish == 0:
                self.active_algo_idx = OWN_COOP_POLICY_IDX
            elif self.n_steps_to_punish > 0:
                self.active_algo_idx = OWN_SELFISH_POLICY_IDX
                self.n_steps_to_punish -= 1
            else:
                raise ValueError("self.n_steps_to_punish can't be below zero")
        else:
            assert self.performing_rollouts
            if self.n_steps_to_punish_opponent == 0:
                self.active_algo_idx = OPP_COOP_POLICY_IDX
            elif self.n_steps_to_punish_opponent > 0:
                self.active_algo_idx = OPP_SELFISH_POLICY_IDX
                self.n_steps_to_punish_opponent -= 1
            else:
                raise ValueError(
                    "self.n_steps_to_punish_opp " "can't be below zero"
                )

    def _on_episode_step(
        self,
        opp_obs,
        last_obs,
        opp_action,
        worker,
        base_env,
        episode,
        env_index,
    ):
        if not self.performing_rollouts:
            super()._on_episode_step(
                opp_obs,
                last_obs,
                opp_action,
                worker,
                base_env,
                episode,
                env_index,
            )

    def _compute_debit(
        self,
        last_obs,
        opp_action,
        worker,
        base_env,
        episode,
        env_index,
        coop_opp_simulated_action,
    ):
        approximated_debit = self._compute_debit_using_rollouts(
            last_obs, opp_action, worker
        )
        return approximated_debit

    def _compute_debit_using_rollouts(self, last_obs, opp_action, worker):
        (
            n_steps_to_punish,
            policy_map,
            policy_agent_mapping,
        ) = self._prepare_to_perform_virtual_rollouts_in_env(worker)

        # Cooperative rollouts
        (
            mean_total_reward_for_totally_coop_opp,
            _,
        ) = self._compute_opp_mean_total_reward(
            worker,
            policy_map,
            policy_agent_mapping,
            partially_coop=False,
            opp_action=None,
            last_obs=last_obs,
        )
        # Cooperative rollouts with first action as the real one
        (
            mean_total_reward_for_partially_coop_opp,
            _,
        ) = self._compute_opp_mean_total_reward(
            worker,
            policy_map,
            policy_agent_mapping,
            partially_coop=True,
            opp_action=opp_action,
            last_obs=last_obs,
        )

        if self.verbose > 0:
            print(
                "r_partially_coop_opp",
                mean_total_reward_for_partially_coop_opp,
                "r_totally_coop_opp",
                mean_total_reward_for_totally_coop_opp,
            )
        opp_reward_gain_from_picking_this_action = (
            mean_total_reward_for_partially_coop_opp
            - mean_total_reward_for_totally_coop_opp
        )

        self._stop_performing_virtual_rollouts_in_env(n_steps_to_punish)

        return opp_reward_gain_from_picking_this_action

    def _prepare_to_perform_virtual_rollouts_in_env(self, worker):
        self.performing_rollouts = True
        self.use_opponent_policies = False
        n_steps_to_punish = self.n_steps_to_punish
        self.n_steps_to_punish = 0
        self.n_steps_to_punish_opponent = 0
        assert self.n_rollout_replicas // 2 > 0
        policy_map = {
            policy_id: self for policy_id in worker.policy_map.keys()
        }
        policy_agent_mapping = lambda agent_id: self._switch_own_and_opp(
            agent_id
        )

        return n_steps_to_punish, policy_map, policy_agent_mapping

    def _stop_performing_virtual_rollouts_in_env(self, n_steps_to_punish):
        self.performing_rollouts = False
        self.use_opponent_policies = False
        self.n_steps_to_punish = n_steps_to_punish

    def _switch_own_and_opp(self, agent_id):
        if agent_id != self.own_policy_id:
            self.use_opponent_policies = True
        else:
            self.use_opponent_policies = False
        return self.own_policy_id

    def _compute_punishment_duration(
        self, opp_action, coop_opp_simulated_action, worker, last_obs
    ):
        return self._compute_punishment_duration_from_rollouts(
            worker, last_obs
        )

    def _compute_punishment_duration_from_rollouts(self, worker, last_obs):
        (
            n_steps_to_punish,
            policy_map,
            policy_agent_mapping,
        ) = self._prepare_to_perform_virtual_rollouts_in_env(worker)

        self.k_opp_loss = {}
        k_to_explore = max(self.last_k, 1)
        k_to_explore = min(
            k_to_explore,
            worker.env.max_steps - worker.env.step_count_in_current_episode,
        )
        k_to_explore = min(k_to_explore, self.rollout_length)
        self.punishment_debit = self.total_debit * self.punishment_multiplier
        self.last_n_steps_played = None
        continue_to_search_k = True
        while continue_to_search_k:
            (
                k_to_explore,
                continue_to_search_k,
            ) = self._search_duration_of_future_punishment(
                k_to_explore,
                worker,
                policy_map,
                policy_agent_mapping,
                last_obs,
            )

        self._stop_performing_virtual_rollouts_in_env(n_steps_to_punish)
        self.last_k = k_to_explore

        print("k_opp_loss", self.k_opp_loss)
        print(
            "k found",
            k_to_explore,
            "self.total_debit",
            self.total_debit,
            "punishment_debit",
            self.punishment_debit,
        )
        return k_to_explore

    def _search_duration_of_future_punishment(
        self, k_to_explore, worker, policy_map, policy_agent_mapping, last_obs
    ):

        self._compute_opp_loss_for_one_k(
            k_to_explore, worker, policy_map, policy_agent_mapping, last_obs
        )
        self._compute_opp_loss_for_one_k(
            k_to_explore - 1,
            worker,
            policy_map,
            policy_agent_mapping,
            last_obs,
        )

        return self._duration_found_or_continue_search(k_to_explore)

    def _duration_found_or_continue_search(self, k_to_explore):
        continue_to_search_k = not self._is_k_found(k_to_explore)

        if continue_to_search_k:
            (
                k_to_explore,
                continue_to_search_k,
            ) = self._stop_search_if_all_small_k_already_explored(k_to_explore)

        if continue_to_search_k:
            (
                k_to_explore,
                continue_to_search_k,
                skip_k_update,
            ) = self._stop_search_if_not_enough_steps_to_be_played(
                k_to_explore
            )

        if continue_to_search_k and not skip_k_update:
            (
                k_to_explore,
                continue_to_search_k,
                skip_other,
            ) = self._if_nothing_remains_to_explore(k_to_explore)

            if continue_to_search_k and not skip_other:
                k_to_explore = self._update_k_to_new_value_to_explore(
                    k_to_explore
                )

        if continue_to_search_k:
            k_to_explore = max(k_to_explore, 1)
        return k_to_explore, continue_to_search_k

    def _k_correction(self, k_to_explore):
        k_to_explore = min(k_to_explore, self.rollout_length)
        k_to_explore = max(k_to_explore, 0)
        return k_to_explore

    def _is_k_found(self, k_to_explore):
        found_k = (
            self.k_opp_loss[k_to_explore] >= self.punishment_debit
            and self.k_opp_loss[k_to_explore - 1] < self.punishment_debit
        )
        return found_k

    def _if_nothing_remains_to_explore(self, k_to_explore):
        continue_to_search_k = True
        skip_other = False

        all_k_explored = self.k_opp_loss.keys()
        all_k_to_explore = list(range(self.rollout_length + 1))
        for k_explored in all_k_explored:
            all_k_to_explore.remove(k_explored)

        if (
            k_to_explore == self.rollout_length
            and not self._is_k_found(k_to_explore)
            and len(all_k_to_explore) > 0
        ):

            k_to_explore = max(all_k_to_explore) + 1
            skip_other = True
            print(
                "explored max k, move to max of the remaining k", k_to_explore
            )

        elif len(all_k_to_explore) == 0:
            continue_to_search_k = False
            if self.k_opp_loss[1] > self.punishment_debit:
                k_to_explore = 1
            elif self.k_opp_loss[self.rollout_length] < self.punishment_debit:
                k_to_explore = self._search_max_punishment()
            print("explored all k use", k_to_explore)
            skip_other = True

        k_to_explore = self._k_correction(k_to_explore)
        return k_to_explore, continue_to_search_k, skip_other

    def _search_max_punishment(self):
        max_v, k_max = -np.inf, self.rollout_length
        for k, v in self.k_opp_loss.items():
            if v > max_v:
                max_v = v
                k_max = k
            elif v == max_v and k < k_max:
                k_max = k
        return k_max

    def _stop_search_if_all_small_k_already_explored(self, k_to_explore):
        continue_to_search_k = True
        if (
            self.k_opp_loss[k_to_explore - 1] > self.punishment_debit
            and (k_to_explore - 1) <= 1
        ):
            k_to_explore = 1
            continue_to_search_k = False
            if self.verbose >= 1:
                print("already explored all the small k", k_to_explore)
        k_to_explore = self._k_correction(k_to_explore)
        return k_to_explore, continue_to_search_k

    def _stop_search_if_not_enough_steps_to_be_played(self, k_to_explore):
        continue_to_search_k = True
        skip_k_update = False
        if self.last_n_steps_played is not None:
            if (
                k_to_explore >= self.last_n_steps_played
                and self.k_opp_loss[k_to_explore] < self.punishment_debit
            ):
                k_to_explore = self._search_max_punishment()
                continue_to_search_k = False
                if self.verbose >= 1:
                    print(
                        "n_steps_played",
                        self.last_n_steps_played,
                        "k_to_explore",
                        k_to_explore,
                    )
            elif k_to_explore > self.last_n_steps_played:
                k_to_explore = self.last_n_steps_played
                skip_k_update = True
                if self.verbose >= 1:
                    print("moving to k = last_n_steps_played", k_to_explore)
        k_to_explore = self._k_correction(k_to_explore)
        return k_to_explore, continue_to_search_k, skip_k_update

    def _update_k_to_new_value_to_explore(self, k_to_explore):
        if self.k_opp_loss[k_to_explore] > self.punishment_debit:
            k_to_explore = min(self.k_opp_loss.keys())
            if self.verbose >= 1:
                print("move toward low k", k_to_explore)
        elif self.k_opp_loss[k_to_explore] < self.punishment_debit:
            k_to_explore = max(self.k_opp_loss.keys()) + 1
            if self.verbose >= 1:
                print("move toward high k", k_to_explore)
        k_to_explore = self._k_correction(k_to_explore)
        return k_to_explore

    def _compute_opp_loss_for_one_k(
        self, k_to_explore, worker, policy_map, policy_agent_mapping, last_obs
    ):
        if self._is_k_out_of_bound(k_to_explore):
            self.k_opp_loss[k_to_explore] = 0
        elif k_to_explore not in self.k_opp_loss.keys():
            (
                opp_total_reward_loss,
                n_steps_played,
            ) = self._compute_opp_total_reward_loss(
                k_to_explore,
                worker,
                policy_map,
                policy_agent_mapping,
                last_obs=last_obs,
            )
            self.k_opp_loss[k_to_explore] = opp_total_reward_loss
            if self.verbose > 0:
                print(f"k_to_explore {k_to_explore}: {opp_total_reward_loss}")

            self.last_n_steps_played = n_steps_played

    def _is_k_out_of_bound(self, k_to_explore):
        return k_to_explore <= 0 or k_to_explore > self.rollout_length

    def _compute_opp_total_reward_loss(
        self, k_to_explore, worker, policy_map, policy_agent_mapping, last_obs
    ):
        # Cooperative rollouts
        (
            coop_mean_total_reward,
            n_steps_played,
        ) = self._compute_opp_mean_total_reward(
            worker,
            policy_map,
            policy_agent_mapping,
            partially_coop=False,
            opp_action=None,
            last_obs=last_obs,
        )
        # Partially cooperative rollouts with first k actions being selfish
        (
            partially_coop_mean_total_reward,
            _,
        ) = self._compute_opp_mean_total_reward(
            worker,
            policy_map,
            policy_agent_mapping,
            partially_coop=False,
            opp_action=None,
            last_obs=last_obs,
            k_to_explore=k_to_explore,
        )

        opp_total_reward_loss = (
            coop_mean_total_reward - partially_coop_mean_total_reward
        )

        if self.verbose > 0:
            print(
                f"partially_coop_mean_total_reward "
                f"{partially_coop_mean_total_reward}"
            )
            print(f"coop_mean_total_reward {coop_mean_total_reward}")
            print(f"opp_total_reward_loss {opp_total_reward_loss}")

        return opp_total_reward_loss, n_steps_played

    def _compute_opp_mean_total_reward(
        self,
        worker,
        policy_map,
        policy_agent_mapping,
        partially_coop: bool,
        opp_action,
        last_obs,
        k_to_explore=0,
    ):
        opp_total_rewards = []
        for i in range(self.n_rollout_replicas // 2):
            self.n_steps_to_punish = k_to_explore
            self.n_steps_to_punish_opponent = k_to_explore
            if partially_coop:
                assert len(self.overwrite_action) == 0
                self.overwrite_action = [
                    (np.array([opp_action]), [], {}),
                ]
            coop_rollout = rollout.internal_rollout(
                worker,
                num_steps=self.rollout_length,
                policy_map=policy_map,
                last_obs=last_obs,
                policy_agent_mapping=policy_agent_mapping,
                reset_env_before=False,
                num_episodes=1,
            )
            assert (
                coop_rollout._num_episodes == 1
            ), f"coop_rollout._num_episodes {coop_rollout._num_episodes}"

            epi = coop_rollout._current_rollout
            opp_rewards = [
                step[3][self.ag_id_rollout_reward_to_read] for step in epi
            ]
            # print("rewards", rewards)
            opp_total_reward = sum(opp_rewards)

            opp_total_rewards.append(opp_total_reward)

            n_steps_played = len(epi)
            assert self.n_steps_to_punish == max(
                0, k_to_explore - n_steps_played
            )
            assert self.n_steps_to_punish_opponent == max(
                0, k_to_explore - n_steps_played
            )
            if n_steps_played > 0:
                assert len(self.overwrite_action) == 0

        self.n_steps_to_punish = 0
        self.n_steps_to_punish_opponent = 0
        opp_mean_total_reward = sum(opp_total_rewards) / len(opp_total_rewards)
        return opp_mean_total_reward, n_steps_played

    def on_episode_end(self, *args, **kwargs):
        if self.working_state == WORKING_STATES[2]:
            self.total_debit = 0
            self.n_steps_to_punish = 0
            self.n_steps_to_punish_opponent = 0
