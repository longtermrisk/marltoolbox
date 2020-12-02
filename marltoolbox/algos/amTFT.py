import logging
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_stats
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import AgentID
from ray.rllib.utils.typing import TensorType
from typing import List, Union, Optional, Dict, Tuple

from marltoolbox.algos import hierarchical
from marltoolbox.utils import preprocessing, log

logger = logging.getLogger(__name__)

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional




APPROXIMATION_METHOD_Q_VALUE = "amTFT_use_Q_net"
APPROXIMATION_METHOD_ROLLOUTS = "amTFT_use_rollout"
APPROXIMATION_METHODS = (APPROXIMATION_METHOD_Q_VALUE, APPROXIMATION_METHOD_ROLLOUTS)
WORKING_STATES = ("train_coop", "train_selfish", "eval_amtft", "eval_naive_selfish", "eval_naive_coop")
WORKING_STATES_IN_EVALUATION = WORKING_STATES[2:]

DEFAULT_CONFIG_UPDATE = merge_dicts(
    hierarchical.HIERARCHICAL_DEFAULT_CONFIG_UPDATE,
    {
        # One of MODES.
        # Set to train_coop to train the cooperative network, to train_selfish to train the selfish and to eval
        "working_state": WORKING_STATES[0],
        "debit_threshold": 10.0,
        "punishement_multiplier": 2.0,
        "mode": APPROXIMATION_METHOD_ROLLOUTS,

        # One from marltoolbox.utils.preprocessing.WELFARES
        "welfare": None,

        'nested_policies': [
            # Here the trainer need to be a DQNTrainer to provide the config for the 3 DQNTorchPolicy
            {"Policy_class":
                 DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats)),
             "config_update": {}},
            {"Policy_class":
                 DQNTorchPolicy.with_updates(stats_fn=log.stats_fn_wt_additionnal_logs(build_q_stats)),
             "config_update": {}},
        ],
        "callbacks": None,
    }
)


class amTFTTorchPolicy(hierarchical.HierarchicalTorchPolicy):
    COOP_POLICY_IDX = 0
    DEFECT_POLICY_IDX = 1

    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)

        self.total_debit = 0
        self.n_steps_to_punish = 0
        self.debit_threshold = config["debit_threshold"]  # T
        self.punishement_multiplier = config["punishement_multiplier"]  # alpha
        self.welfare = config["welfare"]
        assert self.welfare in preprocessing.WELFARES
        self.approximation_method = config["approximation_method"]
        assert self.approximation_method in APPROXIMATION_METHODS

        if self.config["working_state"] in WORKING_STATES_IN_EVALUATION:
            for algo in self.algorithms:
                algo.model.eval()

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


        if self.config["working_state"] == WORKING_STATES[0]:
            actions, state_out, extra_fetches = self.algorithms[self.COOP_POLICY_IDX].compute_actions(obs_batch)
            self.last_used_algo = self.COOP_POLICY_IDX
        elif self.config["working_state"] == WORKING_STATES[1]:
            actions, state_out, extra_fetches = self.algorithms[self.DEFECT_POLICY_IDX].compute_actions(obs_batch)
            self.last_used_algo = self.COOP_POLICY_IDX
        elif self.config["working_state"] == WORKING_STATES[2]:
            if self.n_steps_to_punish == 0:
                actions, state_out, extra_fetches = self.algorithms[self.COOP_POLICY_IDX].compute_actions(obs_batch)
                self.last_used_algo = self.COOP_POLICY_IDX
            elif self.n_steps_to_punish > 0:
                actions, state_out, extra_fetches = self.algorithms[self.DEFECT_POLICY_IDX].compute_actions(obs_batch)
                self.last_used_algo = self.DEFECT_POLICY_IDX
                self.n_steps_to_punish -= 1
            else:
                raise ValueError("self.n_steps_to_punish can't be below zero")
        elif self.config["working_state"] == WORKING_STATES[3]:
            actions, state_out, extra_fetches = self.algorithms[self.DEFECT_POLICY_IDX].compute_actions(obs_batch)
            self.last_used_algo = self.DEFECT_POLICY_IDX
        elif self.config["working_state"] == WORKING_STATES[4]:
            actions, state_out, extra_fetches = self.algorithms[self.COOP_POLICY_IDX].compute_actions(obs_batch)
            self.last_used_algo = self.DEFECT_POLICY_IDX
        else:
            raise ValueError(f'config["working_state"] must be one of {WORKING_STATES}')
        return actions, state_out, extra_fetches

    def learn_on_batch(self, samples: SampleBatch):
        learner_stats = {"learner_stats": {}}

        working_state_idx = WORKING_STATES.index(self.config["working_state"])

        assert working_state_idx == self.COOP_POLICY_IDX or working_state_idx == self.DEFECT_POLICY_IDX

        # Update LR used in optimizer
        self.optimizer()

        samples_copy = samples.copy()
        algo_to_train = self.algorithms[working_state_idx]
        if working_state_idx == self.COOP_POLICY_IDX:
                # TODO use the RLLib batch format directly
                samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[self.welfare])

        # Log true lr
        # TODO Here it is only logging the LR for the 1st parameter
        for j, opt in enumerate(algo_to_train._optimizers):
            self.to_log[f"algo_{working_state_idx}_{j}_lr"] = [p["lr"] for p in opt.param_groups][0]

        learner_stats["learner_stats"][f"learner_stats_algo{working_state_idx}"] = algo_to_train.learn_on_batch(samples_copy)
        self.to_log[f'algo{working_state_idx}_cur_lr'] = algo_to_train.cur_lr

        return learner_stats

    def on_episode_step(self, opp_obs, opp_action, worker, base_env, episode, env_index):
        # If actually using amTFT
        if self.config["working_state"] == WORKING_STATES[2]:
            coop_opp_simulated_action, _, coop_extra_fetches = self.algorithms[self.COOP_POLICY_IDX].compute_actions(
                [opp_obs])
            assert len(coop_extra_fetches["q_values"]) == 1, "batch size need to be 1"
            coop_opp_simulated_action = coop_opp_simulated_action[0]  # Returned lists
            # print("coop_extra_fetches", coop_extra_fetches)
            # print("coop_opp_simulated_action != opp_action", coop_opp_simulated_action, opp_action)
            if coop_opp_simulated_action != opp_action:
                debit, selfish_extra_fetches = self._compute_debit(opp_obs, opp_action, coop_opp_simulated_action)
            else:
                debit = 0
            self.total_debit += debit

            if self.total_debit > self.debit_threshold:
                self.n_steps_to_punish = self._compute_punishment_duration(coop_extra_fetches["q_values"],
                                                                        selfish_extra_fetches["q_values"])
                self.total_debit = 0
            self.to_log['summed_n_steps_to_punish'] = (
                self.n_steps_to_punish + self.to_log['summed_n_steps_to_punish']
                if 'summed_n_steps_to_punish' in self.to_log else self.n_steps_to_punish
            )
            self.to_log['summed_debit'] = (
                debit + self.to_log['summed_debit'] if 'summed_debit' in self.to_log else debit
            )
            self.to_log['debit_threshold'] = self.debit_threshold

    def _compute_debit(self, opp_obs, opp_action, coop_opp_simulated_action):
        if self.approximation_method == APPROXIMATION_METHOD_Q_VALUE:
            return self._compute_debit_from_q_values(opp_obs, opp_action, coop_opp_simulated_action)
        elif self.approximation_method == APPROXIMATION_METHOD_ROLLOUTS:
            return self._compute_debit_from_rollouts(opp_obs, opp_action, coop_opp_simulated_action)
        else:
            raise ValueError(f"self.approximation_method: {self.approximation_method}")

    def _compute_debit_from_q_values(self, opp_obs, opp_action, coop_opp_simulated_action):
        # TODO this is only going to work for symmetrical environments and policies!
        #  (Since I use the agent Q-networks for the opponent)
        opp_simulated_action, _, selfish_extra_fetches = self.algorithms[self.DEFECT_POLICY_IDX].compute_actions(
            [opp_obs])
        opp_simulated_action = opp_simulated_action[0]  # Returned lists
        if opp_simulated_action != opp_action:
            if not self.config["explore"]:
                logger.warning("simulation of opponent not going well since opp_simulated_action != opp_a: "
                               f"{opp_simulated_action}, {opp_action}")
        # TODO solve problem with Q-value changed by exploration => Done?
        temperature_used_for_exploration = self.algorithms[self.DEFECT_POLICY_IDX].exploration.temperature

        debit = (selfish_extra_fetches["q_values"][0, opp_action]*temperature_used_for_exploration -
                 selfish_extra_fetches["q_values"][0, coop_opp_simulated_action]*temperature_used_for_exploration)
        self.to_log['raw_debit'] = debit
        if debit < 0:
            logger.warning(f"debit evaluation not going well since compute debit for this step is: {debit} < 0")
            return 0, selfish_extra_fetches
        else:
            return debit, selfish_extra_fetches

    def _compute_debit_from_rollouts(self):

        pass

    def _compute_punishment_duration(self, q_coop, q_selfish):
        if self.approximation_method == APPROXIMATION_METHOD_Q_VALUE:
            return self._compute_punishment_duration_from_q_values(q_coop, q_selfish)
        elif self.approximation_method == APPROXIMATION_METHOD_ROLLOUTS:
            return self._compute_punishment_duration_from_rollouts()
        else:
            raise ValueError(f"self.approximation_method: {self.approximation_method}")

    def _compute_punishment_duration_from_q_values(self, q_coop, q_selfish):
        # TODO I need to use rollout to make it works like in the paper
        print("q_coop", q_coop)
        print("q_selfish", q_selfish)
        # opp_expected_lost_per_step = q_coop.max() - q_selfish.min()
        opp_expected_lost_per_step = q_selfish.max() - q_selfish.min()
        print("self.total_debit", self.total_debit)
        print("opp_expected_lost_per_step", opp_expected_lost_per_step)
        n_steps_equivalent = (self.total_debit * self.punishement_multiplier) / opp_expected_lost_per_step
        return int(n_steps_equivalent + 1 - 1e-6)

    def _compute_punishment_duration_from_rollouts(self):
        pass

    def on_episode_end(self):
        if self.config["working_state"] == WORKING_STATES[2]:
            self.total_debit = 0
            self.n_steps_to_punish = 0




class amTFTCallBacks(preprocessing.WelfareAndPostprocessCallbacks):

    ADD_UTILITARIAN_WELFARE = True
    ADD_INEQUITY_AVERSION_WELFARE = True

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
        super().on_episode_step(worker=worker, base_env=base_env, episode=episode,
                               env_index=env_index, **kwargs)

        agent_ids = list(worker.policy_map.keys())
        assert len(agent_ids) == 2, "Implemented for 2 players"
        for agent_id, policy in worker.policy_map.items():
            opp_agent_id = [one_id for one_id in agent_ids if one_id != agent_id][0]
            if hasattr(policy, 'on_episode_step') and callable(policy.on_episode_step):
                opp_obs = episode.last_observation_for(opp_agent_id)
                opp_a = episode.last_action_for(opp_agent_id)
                policy.on_episode_step(opp_obs, opp_a, worker, base_env, episode, env_index)

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
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode,
                               env_index=env_index, **kwargs)

        for polidy_ID, policy in policies.items():
            if hasattr(policy, 'on_episode_end') and callable(policy.on_episode_end):
                policy.on_episode_end()



amTFTTrainer = DQNTrainer.with_updates()
