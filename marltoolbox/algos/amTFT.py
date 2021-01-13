from collections import Iterable

import logging
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_stats, postprocess_nstep_and_prio
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.typing import TensorType
from typing import List, Union, Optional, Dict, Tuple

from marltoolbox.algos import hierarchical
from marltoolbox.utils import postprocessing, log, rollout, miscellaneous, restore

logger = logging.getLogger(__name__)

APPROXIMATION_METHOD_Q_VALUE = "amTFT_use_Q_net"
APPROXIMATION_METHOD_ROLLOUTS = "amTFT_use_rollout"
APPROXIMATION_METHODS = (APPROXIMATION_METHOD_Q_VALUE, APPROXIMATION_METHOD_ROLLOUTS)
WORKING_STATES = ("train_coop", "train_selfish", "eval_amtft", "eval_naive_selfish", "eval_naive_coop")
WORKING_STATES_IN_EVALUATION = WORKING_STATES[2:]

OWN_COOP_POLICY_IDX = 0
OWN_SELFISH_POLICY_IDX = 1
OPP_COOP_POLICY_IDX = 2
OPP_SELFISH_POLICY_IDX = 3

DEFAULT_NESTED_POLICY_SELFISH = DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats))
DEFAULT_NESTED_POLICY_COOP = DEFAULT_NESTED_POLICY_SELFISH.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.get_postprocessing_welfare_function(add_utilitarian_welfare=True,),
            postprocess_nstep_and_prio
        )
    )

DEFAULT_CONFIG = merge_dicts(
    hierarchical.DEFAULT_CONFIG,
    {
        # One of WORKING_STATES.
        "working_state": WORKING_STATES[0],
        "debit_threshold": 2.0,
        "punishment_multiplier": 6.0,
        "approximation_method": APPROXIMATION_METHOD_ROLLOUTS,
        "rollout_length": 40,
        "n_rollout_replicas": 20,
        # TODO use log level of RLLib instead of mine
        "verbose": 1,

        # To configure
        "own_policy_id": None,
        "opp_policy_id": None,
        "callbacks": None,
        # One from marltoolbox.utils.postprocessing.WELFARES
        "welfare": None,

        'nested_policies': [
            # Here the trainer need to be a DQNTrainer to provide the config for the 3 DQNTorchPolicy
            {"Policy_class":DEFAULT_NESTED_POLICY_COOP,
             "config_update": {}},
            {"Policy_class":DEFAULT_NESTED_POLICY_SELFISH,
             "config_update": {}},
            {"Policy_class":DEFAULT_NESTED_POLICY_COOP,
             "config_update": {}},
            {"Policy_class":DEFAULT_NESTED_POLICY_SELFISH,
             "config_update": {}},
        ],
    }
)


class amTFTTorchPolicy(hierarchical.HierarchicalTorchPolicy):

    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)

        # Specific to amTFT
        self.total_debit = 0
        self.n_steps_to_punish = 0
        self.debit_threshold = config["debit_threshold"]  # T
        self.punishment_multiplier = config["punishment_multiplier"]  # alpha
        self.welfare = config["welfare"]
        self.working_state = config["working_state"]
        self.verbose = config["verbose"]

        assert self.welfare in postprocessing.WELFARES, f"self.welfare: {self.welfare} must be in " \
                                                        f"postprocessing.WELFARES: {postprocessing.WELFARES}"
        self.approximation_method = config["approximation_method"]
        assert self.approximation_method in APPROXIMATION_METHODS

        # Only used for the rollouts
        self.last_k = 1
        self.use_opponent_policies = False
        self.rollout_length = config["rollout_length"]
        self.n_rollout_replicas = config["n_rollout_replicas"]
        self.performing_rollouts = False
        self.overwrite_action = []
        self.own_policy_id = config["own_policy_id"]
        self.opp_policy_id = config["opp_policy_id"]
        self.n_steps_to_punish_opp = 0

        if self.working_state in WORKING_STATES_IN_EVALUATION:
            for algo in self.algorithms:
                algo.model.eval()

        restore.after_init_load_policy_checkpoint(self)

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

        # Option to overwrite action during internal rollouts
        if self.use_opponent_policies:
            if len(self.overwrite_action) > 0:
                actions, state_out, extra_fetches = self.overwrite_action.pop(0)
                # print("overwritten actions", actions, type(actions))
                return actions, state_out, extra_fetches

        # Select algo tu use
        if self.working_state == WORKING_STATES[0]:
            self.active_algo_idx = OWN_COOP_POLICY_IDX
        elif self.working_state == WORKING_STATES[1]:
            self.active_algo_idx = OWN_SELFISH_POLICY_IDX
        elif self.working_state == WORKING_STATES[2]:
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
                if self.n_steps_to_punish_opp == 0:
                    self.active_algo_idx = OPP_COOP_POLICY_IDX
                elif self.n_steps_to_punish_opp > 0:
                    self.active_algo_idx = OPP_SELFISH_POLICY_IDX
                    self.n_steps_to_punish_opp -= 1
                else:
                    raise ValueError("self.n_steps_to_punish_opp can't be below zero")
        elif self.working_state == WORKING_STATES[3]:
            self.active_algo_idx = OWN_SELFISH_POLICY_IDX
        elif self.working_state == WORKING_STATES[4]:
            self.active_algo_idx = OWN_COOP_POLICY_IDX
        else:
            raise ValueError(f'config["working_state"] must be one of {WORKING_STATES}')

        if self.verbose > 2:
            print(f"self.active_algo_idx {self.active_algo_idx}")
        actions, state_out, extra_fetches = self.algorithms[self.active_algo_idx].compute_actions(obs_batch,
                                                                                                  state_batches,
                                                                                                  prev_action_batch,
                                                                                                  prev_reward_batch,
                                                                                                  info_batch,
                                                                                                  episodes,
                                                                                                  explore,
                                                                                                  timestep,
                                                                                                  **kwargs)
        return actions, state_out, extra_fetches

    def learn_on_batch(self, samples: SampleBatch):
        learner_stats = {"learner_stats": {}}

        working_state_idx = WORKING_STATES.index(self.working_state)

        assert working_state_idx == OWN_COOP_POLICY_IDX or working_state_idx == OWN_SELFISH_POLICY_IDX

        # Update LR used in optimizer
        self.optimizer()

        samples_copy = samples.copy()
        algo_to_train = self.algorithms[working_state_idx]

        # if working_state_idx == OWN_COOP_POLICY_IDX:
        #     # TODO use the RLLib batch format directly
        #     samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[self.welfare])
        # print("working_state_idx", working_state_idx)
        # print("samples_copy[samples_copy.REWARDS]", samples_copy[samples_copy.REWARDS])
        # Log true lr
        # TODO Here it is only logging the LR for the 1st parameter
        for j, opt in enumerate(algo_to_train._optimizers):
            self.to_log[f"algo_{working_state_idx}_{j}_lr"] = [p["lr"] for p in opt.param_groups][0]

        learner_stats["learner_stats"][f"learner_stats_algo{working_state_idx}"] = algo_to_train.learn_on_batch(
            samples_copy)
        self.to_log[f'algo{working_state_idx}_cur_lr'] = algo_to_train.cur_lr

        if self.verbose > 1:
            print(f"learn_on_batch WORKING_STATES {WORKING_STATES[working_state_idx]}, ")

        return learner_stats

    def on_episode_step(self, opp_obs, last_obs, opp_action, worker, base_env, episode, env_index):
        # If actually using amTFT
        # if self.verbose > 2:
        #     print(f"on_episode_step performing_rollouts {self.performing_rollouts}")

        if self.working_state == WORKING_STATES[2] and not self.performing_rollouts:

            if self.n_steps_to_punish == 0:
                coop_opp_simulated_action, _, coop_extra_fetches = self.algorithms[
                    OWN_COOP_POLICY_IDX].compute_actions(
                    [opp_obs])
                coop_opp_simulated_action = coop_opp_simulated_action[0]  # Returned lists
                # print("coop_extra_fetches", coop_extra_fetches)
                # print("coop_opp_simulated_action != opp_action", coop_opp_simulated_action, opp_action)
                if coop_opp_simulated_action != opp_action:
                    if self.verbose > 0:
                        print("coop_opp_simulated_action != opp_action:", coop_opp_simulated_action, opp_action)
                    debit, selfish_extra_fetches = self._compute_debit(opp_obs, last_obs, opp_action,
                                                                       coop_opp_simulated_action,
                                                                       worker, base_env, episode, env_index)
                else:
                    if self.verbose > 0:
                        print("coop_opp_simulated_action == opp_action")
                    debit = 0
                self.total_debit += debit
                self.to_log['summed_debit'] = (
                    debit + self.to_log['summed_debit'] if 'summed_debit' in self.to_log else debit
                )
                if self.verbose > 0:
                    print(f"self.total_debit {self.total_debit}")
            else:
                if self.verbose > 0:
                    print(f"punishing self.n_steps_to_punish: {self.n_steps_to_punish}")
            if self.total_debit > self.debit_threshold:
                self.n_steps_to_punish = self._compute_punishment_duration(coop_extra_fetches["q_values"],
                                                                           selfish_extra_fetches["q_values"],
                                                                           opp_action,
                                                                           coop_opp_simulated_action,
                                                                           worker, last_obs)
                self.total_debit = 0
                self.to_log['summed_n_steps_to_punish'] = (
                    self.n_steps_to_punish + self.to_log['summed_n_steps_to_punish']
                    if 'summed_n_steps_to_punish' in self.to_log else self.n_steps_to_punish
                )

            self.to_log['debit_threshold'] = self.debit_threshold
            print('debit_threshold', self.debit_threshold)

    def _compute_debit(self, opp_obs, last_obs, opp_action, coop_opp_simulated_action,
                       worker, base_env, episode, env_index):
        opp_simu_coop_action, _, selfish_extra_fetches = self.algorithms[OPP_COOP_POLICY_IDX].compute_actions(
            [opp_obs])
        opp_simu_coop_action = opp_simu_coop_action[0]  # Returned lists
        if opp_simu_coop_action == opp_action:
            approximated_debit = 0.0
        else:
            if self.approximation_method == APPROXIMATION_METHOD_Q_VALUE:
                approximated_debit = self._compute_debit_from_q_values(opp_action, opp_simu_coop_action,
                                                                       coop_opp_simulated_action, selfish_extra_fetches)
            elif self.approximation_method == APPROXIMATION_METHOD_ROLLOUTS:
                approximated_debit = self._compute_debit_from_rollouts(last_obs, opp_action, coop_opp_simulated_action,
                                                                       worker, base_env, episode, env_index)
            else:
                raise ValueError(f"self.approximation_method: {self.approximation_method}")

        if self.verbose > 0:
            print(f"approximated_debit {approximated_debit}")
        return approximated_debit, selfish_extra_fetches

    def _compute_debit_from_q_values(self, opp_action, opp_simulated_action, coop_opp_simulated_action,
                                     selfish_extra_fetches):
        # TODO this is only going to work for symmetrical environments and policies!
        #  (Since I use the agent Q-networks for the opponent)
        assert len(selfish_extra_fetches["q_values"]) == 1, "batch size need to be 1"

        if not self.config["explore"]:
            logger.warning("simulation of opponent not going well since opp_simulated_action != opp_a: "
                           f"{opp_simulated_action}, {opp_action}")
        # TODO solve problem with Q-value changed by exploration => Done?
        temperature_used_for_exploration = self.algorithms[OWN_SELFISH_POLICY_IDX].exploration.temperature

        debit = (selfish_extra_fetches["q_values"][0, opp_action] * temperature_used_for_exploration -
                 selfish_extra_fetches["q_values"][0, coop_opp_simulated_action] * temperature_used_for_exploration)
        self.to_log['raw_debit'] = debit
        return debit

        # if debit < 0:
        #     logger.warning(f"debit evaluation not going well since compute debit for this step is: {debit} < 0")
        #     return 0
        # else:
        #     return debit

    def _switch_own_and_opp(self, agent_id):
        if agent_id != self.own_policy_id:
            self.use_opponent_policies = True
        else:
            self.use_opponent_policies = False
        return self.own_policy_id

    def _compute_opp_mean_total_reward(self, worker, policy_map, policy_agent_mapping, partially_coop: bool,
                                       opp_action, last_obs, k_to_explore=0):
        total_rewards = []
        for i in range(self.n_rollout_replicas // 2):
            self.n_steps_to_punish = k_to_explore
            self.n_steps_to_punish_opp = k_to_explore
            if partially_coop:
                assert len(self.overwrite_action) == 0
                self.overwrite_action = [(np.array([opp_action]), [], {}), ]
            coop_rollout = rollout.internal_rollout(worker,
                                                    num_steps=self.rollout_length,
                                                    policy_map=policy_map,
                                                    last_obs=last_obs,
                                                    policy_agent_mapping=policy_agent_mapping,
                                                    reset_env_before=False,
                                                    num_episodes=1)
            assert coop_rollout._num_episodes == 1, f"coop_rollout._num_episodes {coop_rollout._num_episodes}"
            epi = coop_rollout._current_rollout
            rewards = [step[3][self.opp_policy_id] for step in epi]
            # print("rewards", rewards)
            total_reward = sum(rewards)

            total_rewards.append(total_reward)
        # print("total_rewards", total_rewards)
        self.n_steps_to_punish = 0
        self.n_steps_to_punish_opp = 0
        n_steps_played = len(epi)
        mean_total_reward = sum(total_rewards) / len(total_rewards)
        return mean_total_reward, n_steps_played

    def _compute_debit_from_rollouts(self, last_obs, opp_action, coop_opp_simulated_action,
                                     worker, base_env, episode, env_index):
        self.performing_rollouts = True
        self.use_opponent_policies = False
        n_steps_to_punish = self.n_steps_to_punish
        self.n_steps_to_punish = 0
        self.n_steps_to_punish_opp = 0
        assert self.n_rollout_replicas // 2 > 0
        policy_map = {policy_id: self for policy_id in worker.policy_map.keys()}
        policy_agent_mapping = (lambda agent_id: self._switch_own_and_opp(agent_id))

        # Cooperative rollouts
        coop_mean_total_reward, _ = self._compute_opp_mean_total_reward(worker, policy_map, policy_agent_mapping,
                                                                        partially_coop=False, opp_action=None,
                                                                        last_obs=last_obs)
        # Cooperative rollouts with first action as the real one
        partially_coop_mean_total_reward, _ = self._compute_opp_mean_total_reward(worker, policy_map,
                                                                                  policy_agent_mapping,
                                                                                  partially_coop=True,
                                                                                  opp_action=opp_action,
                                                                                  last_obs=last_obs)

        print("partially_coop_mean_total_reward", partially_coop_mean_total_reward)
        print("coop_mean_total_reward", coop_mean_total_reward)
        opp_total_reward_gain = partially_coop_mean_total_reward - coop_mean_total_reward

        self.performing_rollouts = False
        self.use_opponent_policies = False
        self.n_steps_to_punish = n_steps_to_punish

        return opp_total_reward_gain
        #
        # if opp_total_reward_gain < 0:
        #     logger.warning(f"debit evaluation not going well since compute debit for this step is: {opp_total_reward_gain} < 0")
        #     return 0
        # else:
        #     return opp_total_reward_gain

    def _compute_punishment_duration(self, q_coop, q_selfish, opp_action, coop_opp_simulated_action, worker, last_obs):
        if self.approximation_method == APPROXIMATION_METHOD_Q_VALUE:
            return self._compute_punishment_duration_from_q_values(q_coop, q_selfish, opp_action,
                                                                   coop_opp_simulated_action)
        elif self.approximation_method == APPROXIMATION_METHOD_ROLLOUTS:
            return self._compute_punishment_duration_from_rollouts(worker, last_obs)
        else:
            raise ValueError(f"self.approximation_method: {self.approximation_method}")

    def _compute_punishment_duration_from_q_values(self, q_coop, q_selfish, opp_action, coop_opp_simulated_action):
        # TODO this is only going to work for symmetrical environments
        # TODO This is only going to work if each action has the same impact on the reward

        print("q_coop", q_coop)
        print("q_selfish", q_selfish)
        # opp_expected_lost_per_step = q_selfish.max() - q_selfish.min()
        opp_expected_lost_per_step = q_coop[coop_opp_simulated_action] / 2 - q_selfish[opp_action]
        print("self.total_debit", self.total_debit)
        print("opp_expected_lost_per_step", opp_expected_lost_per_step)
        n_steps_equivalent = (self.total_debit * self.punishment_multiplier) / opp_expected_lost_per_step
        return int(n_steps_equivalent + 1 - 1e-6)

    def _compute_opp_total_reward_loss(self, k_to_explore, worker, policy_map, policy_agent_mapping, last_obs):
        # Cooperative rollouts
        # print("start Cooperative rollouts")
        coop_mean_total_reward, n_steps_played = self._compute_opp_mean_total_reward(worker, policy_map,
                                                                                     policy_agent_mapping,
                                                                                     partially_coop=False,
                                                                                     opp_action=None,
                                                                                     last_obs=last_obs)
        # Cooperative rollouts with first action as the real one
        # print("start Partially cooperative rollouts")
        partially_coop_mean_total_reward, _ = self._compute_opp_mean_total_reward(worker, policy_map,
                                                                                  policy_agent_mapping,
                                                                                  partially_coop=False,
                                                                                  opp_action=None, last_obs=last_obs,
                                                                                  k_to_explore=k_to_explore)

        opp_total_reward_loss = coop_mean_total_reward - partially_coop_mean_total_reward

        # if self.verbose > 0:
        #     print(f"partially_coop_mean_total_reward {partially_coop_mean_total_reward}")
        #     print(f"coop_mean_total_reward {coop_mean_total_reward}")
        #     print(f"opp_total_reward_loss {opp_total_reward_loss}")

        return opp_total_reward_loss, n_steps_played

    def _compute_punishment_duration_from_rollouts(self, worker, last_obs):
        self.performing_rollouts = True
        self.use_opponent_policies = False
        n_steps_to_punish = self.n_steps_to_punish
        assert self.n_rollout_replicas // 2 > 0
        policy_map = {policy_id: self for policy_id in worker.policy_map.keys()}
        policy_agent_mapping = (lambda agent_id: self._switch_own_and_opp(agent_id))

        k_opp_loss = {}
        k_to_explore = self.last_k

        debit_threshold = self.total_debit * self.punishment_multiplier

        continue_to_search_k = True
        # print("start searching k, total_debit:", self.total_debit)
        while continue_to_search_k:
            # Compute for k
            if k_to_explore <= 0:
                k_opp_loss[k_to_explore] = -999
            elif k_to_explore not in k_opp_loss.keys():
                opp_total_reward_loss, n_steps_played = self._compute_opp_total_reward_loss(k_to_explore, worker,
                                                                                            policy_map,
                                                                                            policy_agent_mapping,
                                                                                            last_obs=last_obs)
                k_opp_loss[k_to_explore] = opp_total_reward_loss
                if self.verbose > 0:
                    print(f"k_to_explore {k_to_explore}: {opp_total_reward_loss}")

            # Compute for (k - 1)
            if (k_to_explore - 1) <= 0:
                k_opp_loss[k_to_explore - 1] = -999
            elif (k_to_explore - 1) not in k_opp_loss.keys():
                opp_total_reward_loss, _ = self._compute_opp_total_reward_loss(k_to_explore - 1, worker, policy_map,
                                                                               policy_agent_mapping, last_obs=last_obs)
                k_opp_loss[k_to_explore - 1] = opp_total_reward_loss
                if self.verbose > 0:
                    print(f"k_to_explore-1 {k_to_explore - 1}: {opp_total_reward_loss}")

            found_k = (k_opp_loss[k_to_explore] >= debit_threshold and
                       k_opp_loss[k_to_explore - 1] <= debit_threshold)
            continue_to_search_k = not found_k

            if not continue_to_search_k:
                break

            # If all the smallest k are already explored
            if (k_opp_loss[k_to_explore - 1] > debit_threshold and (k_to_explore - 1) <= 1):
                k_to_explore = 1
                break

            # If there is not enough steps to be perform remaining in the episode
            # to compensate for the current total debit
            if k_to_explore >= n_steps_played and k_opp_loss[k_to_explore] < debit_threshold:
                print("n_steps_played", n_steps_played, "k_to_explore", k_to_explore)
                k_to_explore = max(k_opp_loss.keys())
                break

            if continue_to_search_k:
                if k_opp_loss[k_to_explore] > debit_threshold:
                    k_to_explore = min(k_opp_loss.keys())
                elif k_opp_loss[k_to_explore] < debit_threshold:
                    k_to_explore = max(k_opp_loss.keys()) + 1

        self.performing_rollouts = False
        self.use_opponent_policies = False
        # Useless since it will be overwritten
        self.n_steps_to_punish = n_steps_to_punish
        self.last_k = k_to_explore

        print("k_opp_loss", k_opp_loss)
        print("k found", k_to_explore, "self.total_debit", self.total_debit, "debit_threshold", debit_threshold)
        return k_to_explore

    def on_episode_end(self):
        if self.working_state == WORKING_STATES[2]:
            self.total_debit = 0
            self.n_steps_to_punish = 0


def two_steps_training(stop, config, name, do_not_load=[], TrainerClass=DQNTrainer, **kwargs):
    for policy_id in config["multiagent"]["policies"].keys():
        config["multiagent"]["policies"][policy_id][3]["working_state"] = "train_selfish"

    results = ray.tune.run(TrainerClass, config=config,
                           stop=stop, name=name,
                           checkpoint_at_end=True,
                           metric="episode_reward_mean", mode="max",
                           **kwargs)
    checkpoints = miscellaneous.extract_checkpoints(results)
    seeds = miscellaneous.extract_config_value(results, "seed")
    seed_to_checkpoint = {}
    for seed, checkpoint in zip(seeds, checkpoints):
        seed_to_checkpoint[seed] = checkpoint

    # Train internal selfish policy
    for policy_id in config["multiagent"]["policies"].keys():
        config["multiagent"]["policies"][policy_id][3]["working_state"] = "train_coop"
        if policy_id not in do_not_load:
            config["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = (
                miscellaneous.seed_to_checkpoint(seed_to_checkpoint), policy_id
            )

    # amTFTTrainerTrainSelfish = restore.prepare_trainer_to_load_checkpoints(amTFTTrainer)
    results = ray.tune.run(TrainerClass, config=config,
                           stop=stop, name=name,
                           checkpoint_at_end=True,
                           metric="episode_reward_mean", mode="max",
                           **kwargs)
    return results


# TODO do the same in postprocessing (closure)
def get_amTFTCallBacks(additionnal_callbacks=[], **kwargs):
    # WelfareAndPostprocessCallbacks = postprocessing.get_WelfareAndPostprocessCallbacks(**kwargs)

    class amTFTCallBacksPart(DefaultCallbacks):

        def on_episode_step(self, *, worker, base_env,
                            episode, env_index, **kwargs):
            agent_ids = list(worker.policy_map.keys())
            assert len(agent_ids) == 2, "Implemented for 2 players"
            for agent_id, policy in worker.policy_map.items():
                opp_agent_id = [one_id for one_id in agent_ids if one_id != agent_id][0]
                if hasattr(policy, 'on_episode_step') and callable(policy.on_episode_step):
                    opp_obs = episode.last_observation_for(opp_agent_id)
                    last_obs = episode._agent_to_last_raw_obs
                    opp_a = episode.last_action_for(opp_agent_id)
                    policy.on_episode_step(opp_obs, last_obs, opp_a, worker, base_env, episode, env_index)

        def on_episode_end(self, *, worker, base_env,
                           policies, episode, env_index, **kwargs):

            for polidy_ID, policy in policies.items():
                if hasattr(policy, 'on_episode_end') and callable(policy.on_episode_end):
                    policy.on_episode_end()

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            self._share_weights_during_training(trainer)

        def _share_weights_during_training(self, trainer):
            local_policy_map = trainer.workers.local_worker().policy_map
            policy_ids = list(local_policy_map.keys())
            assert len(policy_ids) == 2
            policies_weights = None
            for i, policy_id in enumerate(policy_ids):
                opp_policy_id = policy_ids[(i + 1) % 2]
                if isinstance(local_policy_map[policy_id], amTFTTorchPolicy):
                    if local_policy_map[policy_id].working_state not in WORKING_STATES_IN_EVALUATION:
                        policy = local_policy_map[policy_id]
                        # Only get and set weights during training
                        if policies_weights is None:
                            policies_weights = trainer.get_weights()
                        assert isinstance(local_policy_map[opp_policy_id],
                                          amTFTTorchPolicy), "if amTFT is training then " \
                                                             "the opponent must be " \
                                                             "using amTFT too"
                        # share weights during training of amTFT
                        policies_weights[policy_id][policy._nested_key(OPP_COOP_POLICY_IDX)] = \
                            policies_weights[opp_policy_id][policy._nested_key(OWN_COOP_POLICY_IDX)]
                        policies_weights[policy_id][policy._nested_key(OPP_SELFISH_POLICY_IDX)] = \
                            policies_weights[opp_policy_id][policy._nested_key(OWN_SELFISH_POLICY_IDX)]
            # Only get and set weights during training
            if policies_weights is not None:
                trainer.set_weights(policies_weights)

    if not isinstance(additionnal_callbacks, Iterable):
        additionnal_callbacks = [additionnal_callbacks]

    amTFTCallBacks = miscellaneous.merge_callbacks(  # WelfareAndPostprocessCallbacks,
        amTFTCallBacksPart,
        *additionnal_callbacks)

    return amTFTCallBacks

# amTFTTrainer = DQNTrainer.with_updates()
