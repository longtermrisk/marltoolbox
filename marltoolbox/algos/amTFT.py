import numpy as np
from typing import List, Union, Optional, Dict, Tuple

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, build_q_stats
from ray.rllib.utils.schedules import PiecewiseSchedule

from ray.rllib.utils.typing import AgentID, PolicyID, TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.agents import dqn
from ray.rllib.agents.callbacks import DefaultCallbacks

from marltoolbox.utils.exploration import SoftQSchedule
from marltoolbox.utils import preprocessing, logging, restore
from marltoolbox.algos import hierarchical

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional

AMTFT_TRAIN_KEY = "amTFT_train"

AMTF_DEFAULT_CONFIG_UPDATE = merge_dicts(
    hierarchical.HIERARCHICAL_DEFAULT_CONFIG_UPDATE,
    {
        # Set to True to train the nested policies and to False to use them
        "amTFT_train": True,
        "debit_threshold": 10.0,
        "punishement_multiplier": 2.0,

        'nested_policies': [
            # Here the trainer need to be a DQNTrainer to provide the config for the 3 DQNTorchPolicy
            {"Policy_class": DQNTorchPolicy.with_updates(stats_fn=logging.stats_fn_wt_additionnal_logs(
                                                            build_q_stats)),
            "config_update": {}},
            {"Policy_class": DQNTorchPolicy.with_updates(stats_fn=logging.stats_fn_wt_additionnal_logs(
                                                            build_q_stats)),
            "config_update": {}},
        ]
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

        if self.config[AMTFT_TRAIN_KEY]:
            actions, state_out, extra_fetches = self.algorithms[self.COOP_POLICY_IDX].compute_actions(obs_batch)
            self.last_used_algo = self.COOP_POLICY_IDX
        else:
            if self.n_steps_to_punish == 0:
                actions, state_out, extra_fetches = self.algorithms[self.COOP_POLICY_IDX].compute_actions(obs_batch)
                self.last_used_algo = self.COOP_POLICY_IDX
            elif self.n_steps_to_punish > 0:
                actions, state_out, extra_fetches = self.algorithms[self.DEFECT_POLICY_IDX].compute_actions(obs_batch)
                self.last_used_algo = self.DEFECT_POLICY_IDX
                self.n_steps_to_punish -= 1
            else:
                raise ValueError("self.n_steps_to_punish can't be below zero")
        return actions, state_out, extra_fetches


    def learn_on_batch(self, samples: SampleBatch):
        learner_stats = {"learner_stats": {}}

        # Update LR used in optimizer
        self.optimizer()

        if self.config[AMTFT_TRAIN_KEY]:
            for i, algo in enumerate(self.algorithms):

                # Log true lr
                # TODO Here it is only logging for the 1st parameter
                for j, opt in enumerate(algo._optimizers):
                    self.to_log[f"algo_{i}_{j}_lr"] = [p["lr"] for p in opt.param_groups][0]

                # TODO use the RLLib batch format directly
                samples_copy = samples.copy()
                if i == self.COOP_POLICY_IDX:
                    samples_copy[samples_copy.REWARDS] = np.array(samples_copy.data[
                                                                      preprocessing.UTILITARIAN_WELFARE_KEY])
                elif i == self.DEFECT_POLICY_IDX:
                    pass

                # TODO currently continue to learn because I use the batch to stop but the batch is sampled!
                learner_stats["learner_stats"][f"learner_stats_algo{i}"] = algo.learn_on_batch(samples_copy)
                self.to_log[f'algo{i}_cur_lr'] = algo.cur_lr

        return learner_stats


    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[
                "Policy", SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        """Implements algorithm-specific trajectory postprocessing.

        This will be called on each trajectory fragment computed during policy
        evaluation. Each fragment is guaranteed to be only from one episode.

        Args:
            sample_batch (SampleBatch): batch of experiences for the policy,
                which will contain at most one episode trajectory.
            other_agent_batches (dict): In a multi-agent env, this contains a
                mapping of agent ids to (policy, agent_batch) tuples
                containing the policy and experiences of the other agents.
            episode (Optional[MultiAgentEpisode]): An optional multi-agent
                episode object to provide access to all of the
                internal episode state, which may be useful for model-based or
                multi-agent algorithms.

        Returns:
            SampleBatch: Postprocessed sample batch.
        """
        return preprocessing.postprocess_trajectory(sample_batch, other_agent_batches, episode,
                                                          add_utilitarian_welfare=True)

    def on_episode_step(self, opp_obs, opp_action):
        if not self.config[AMTFT_TRAIN_KEY]:
            coop_opp_simulated_action, _, coop_extra_fetches = self.algorithms[self.COOP_POLICY_IDX].compute_actions(
                [opp_obs])
            coop_opp_simulated_action = coop_opp_simulated_action[0]  # Returned lists
            # print("coop_extra_fetches", coop_extra_fetches)
            # print("coop_opp_simulated_action != opp_action", coop_opp_simulated_action, opp_action)
            if coop_opp_simulated_action != opp_action:
                debit, selfish_extra_fetches = self._compute_debit(opp_obs, opp_action, coop_opp_simulated_action)
            else:
                debit = 0
            self.total_debit += debit

            if self.total_debit > self.debit_threshold:
                self.n_steps_to_punish = self._compute_n_step_to_punish(coop_extra_fetches["q_values"], selfish_extra_fetches["q_values"])
                self.total_debit = 0
            self.to_log['debit'] = debit
            self.to_log['total_debit'] = self.total_debit
            self.to_log['n_steps_to_punish'] = self.n_steps_to_punish

    def _compute_debit(self, opp_obs, opp_action, coop_opp_simulated_action):
        # TODO this is only going to work for symmetrical environments and policies!
        opp_simulated_action, _, selfish_extra_fetches = self.algorithms[self.DEFECT_POLICY_IDX].compute_actions([opp_obs])
        opp_simulated_action = opp_simulated_action[0]  # Returned lists
        if opp_simulated_action != opp_action:
            print("simulation of opponent not going well since opp_simulated_action != opp_a:",
                  opp_simulated_action, opp_action)
        debit = selfish_extra_fetches["q_values"][opp_action] - selfish_extra_fetches["q_values"][coop_opp_simulated_action]
        if debit < 0:
            print("debit evaluation not going well since debit < 0:", debit)
            return 0, selfish_extra_fetches
        else:
            return debit, selfish_extra_fetches

    def _compute_n_step_to_punish(self, q_coop, q_selfish):
        # TODO I need to use rollout to make it works like in the paper
        opp_expected_lost_per_step = max(q_coop) - min(q_selfish)
        n_steps_equivalent = (self.total_debit * self.punishement_multiplier) / opp_expected_lost_per_step
        return int(n_steps_equivalent + 1 - 1e-6)

    def on_episode_end(self):
        if not self.config[AMTFT_TRAIN_KEY]:
            self.total_debit = 0
            self.n_steps_to_punish = 0



class amTFTCallBacks(DefaultCallbacks):

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

                opp_obs = episode.last_observation_for(opp_agent_id)
                opp_a = episode.last_action_for(opp_agent_id)
                # if episode.length > 0:
                #     being_punished_by_LE = episode.last_pi_info_for(opp_agent_id).get("punishing", False)
                #     policy.on_episode_step(opp_obs, opp_a, being_punished_by_LE)
                # else:
                #     Only during init
                #     policy.on_episode_step(opp_obs, opp_a, False)
                policy.on_episode_step(opp_obs, opp_a)

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

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        logging.update_train_result_wt_to_log(trainer, result)

amTFTTrainer = DQNTrainer.with_updates(

)