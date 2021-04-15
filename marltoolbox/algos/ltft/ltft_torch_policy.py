##########
# Code modified from:
# https://github.com/Manuscrit/SLM-Lab/blob/
# support_multi_agents_with_awareness/slm_lab/
# agent/algorithm/meta_algorithm/learning_equilibrium.py
##########

import copy
import logging
from collections import deque
from typing import Optional, Dict, Tuple, TYPE_CHECKING, Union, List

import numpy as np
import torch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AgentID, PolicyID, TensorType
from ray.rllib.utils.torch_ops import convert_to_torch_tensor

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

from marltoolbox.algos import hierarchical
from marltoolbox.utils.miscellaneous import filter_sample_batch
from marltoolbox.utils import callbacks, postprocessing

logger = logging.getLogger(__name__)

LTFT_DEFAULT_CONFIG_UPDATE = merge_dicts(
    hierarchical.DEFAULT_CONFIG,
    {
        "percentile_for_likelihood_test": 95,
        "punishment_time": 1,
        "min_coop_epi_after_punishing": 0,
        "defection_threshold": 0.0,
        "average_defection_value": True,
        "average_defection_value_len": 20,
        "use_last_step_for_search": True,
        "length_of_history": 200,
        "n_steps_in_bootstrap_replicates": 20,
        "n_bootstrap_replicates": 50,
    },
)

BEING_PUNISHED = "being_punished"
PUNISHING = "punishing"


class LTFTTorchPolicy(hierarchical.HierarchicalTorchPolicy):
    """
    Learning Tit-for-tat (LTFT) policy
    """

    EPSILON = 1e-12
    DEBUG = False

    COOP_POLICY_IDX = 0
    PUNITIVE_POLICY_IDX = 1
    COOP_OPP_POLICY_IDX = 2
    SPL_OPP_POLICY_IDX = 3

    WARMUP_LENGTH = 0
    INITIALLY_ACTIVE_ALGO = COOP_POLICY_IDX

    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            config,
            **kwargs,
        )

        self.percentile_for_likelihood_test = config[
            "percentile_for_likelihood_test"
        ]
        self.punishment_time = config["punishment_time"]
        self.min_coop_epi_after_punishing = config[
            "min_coop_epi_after_punishing"
        ]
        self.defection_threshold = config["defection_threshold"]
        self.average_defection_value = (config["average_defection_value"],)
        self.average_defection_value_len = config[
            "average_defection_value_len"
        ]
        self.use_last_step_for_search = config["use_last_step_for_search"]
        self.length_of_history = config["length_of_history"]
        self.n_steps_in_bootstrap_replicates = config[
            "n_steps_in_bootstrap_replicates"
        ]
        self.n_bootstrap_replicates = config["n_bootstrap_replicates"]

        assert len(self.algorithms) == 4, str(len(self.algorithms))
        self.opp_policy_from_supervised_learning = True

        # To coordinate the learning between the nested policies
        self.train_pg = True
        self.train_dqn = True

        self.n_steps_since_start = 0
        self.last_computed_w = None
        self.opp_previous_obs = None
        self.opp_new_obs = None
        self.observed_n_step_in_current_epi = 0

        self.data_queue = deque(maxlen=self.length_of_history)

        # Defection
        self.detected_defection = False
        self.defection_metric = 0
        if not self.average_defection_value:
            self.average_defection_value_len = 1
        self.defection_carac_queue = deque(
            maxlen=self.average_defection_value_len
        )
        self.remaining_punishing_time = 0
        self.being_punished_by_opp = False

        self._reset_learner_stats()

        # Logging
        log_len_in_steps = 100
        # for 0) the cooperative policy
        self.action_pd_coop = deque(maxlen=log_len_in_steps)
        # for 1) the punitive policy
        self.action_pd_punish = deque(maxlen=log_len_in_steps)
        # for 2) the estimated opponent cooperative policy
        self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
        # for 3) the estimated opponent policy
        self.action_pd_opp = deque(maxlen=log_len_in_steps)
        self.n_cooperation_steps_in_current_epi = 0
        self.n_punishment_steps_in_current_epi = 0

        self.add_welfare_fn = postprocessing.welfares_postprocessing_fn(
            add_utilitarian_welfare=True,
            add_opponent_action=True,
            add_opponent_neg_reward=True,
        )

        self.view_requirements.update(
            {
                PUNISHING: ViewRequirement(),
            }
        )

    def _reset_learner_stats(self):
        self.learner_stats = {"learner_stats": {}}

    @override(hierarchical.HierarchicalTorchPolicy)
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        obs_batch = input_dict["obs"]
        assert len(obs_batch) == 1, (
            "The LTFT policy only works when sampling/infering one step "
            "at a time."
        )

        actions, state_out, extra_fetches = super()._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )
        extra_fetches[PUNISHING] = [
            self._is_punishing() for _ in obs_batch.tolist()
        ]

        return actions, state_out, extra_fetches

    def _is_punishing(self):
        return self.active_algo_idx == self.PUNITIVE_POLICY_IDX

    @override(hierarchical.HierarchicalTorchPolicy)
    def _learn_on_batch(self, samples: SampleBatch):
        (
            policies_idx_to_train,
            policies_to_train,
        ) = self._get_policies_idx_to_train_with_current_batch()
        if len(policies_idx_to_train) == 0:
            return self.learner_stats
        logging.debug(f"policies_idx_to_train {policies_idx_to_train}")
        self._init_log_learn_on_batch(policies_idx_to_train)

        for policy_n, algo in zip(policies_idx_to_train, policies_to_train):
            samples_copy = samples.copy()
            samples_copy = self._modify_batch_for_policy(
                policy_n, samples_copy
            )

            if (
                len(samples_copy[samples_copy.ACTIONS])
                > len(samples[samples.ACTIONS]) // 2
            ):
                self._to_log[f"learn_on_batch_algo{policy_n}"] = len(
                    samples_copy[samples_copy.ACTIONS]
                )
                self.learner_stats["learner_stats"][
                    f"algo{policy_n}"
                ] = algo.learn_on_batch(samples_copy)
            else:
                self.learner_stats["learner_stats"][f"algo{policy_n}"] = {}

        return self.learner_stats

    def on_train_result(self, *args, **kwargs):
        self._reset_learner_stats()

    def _get_policies_idx_to_train_with_current_batch(self):
        nested_policies_to_train = []
        if self.train_dqn:
            nested_policies_to_train.extend([0, 1, 2])
        if self.train_pg:
            nested_policies_to_train.append(3)

        policy_to_train = [
            self.algorithms[algo_idx] for algo_idx in nested_policies_to_train
        ]

        return nested_policies_to_train, policy_to_train

    def _modify_batch_for_policy(self, policy_n, samples_copy):
        if (
            policy_n == self.COOP_POLICY_IDX
            or policy_n == self.COOP_OPP_POLICY_IDX
        ):
            samples_copy[samples_copy.REWARDS] = np.array(
                samples_copy.data[postprocessing.WELFARE_UTILITARIAN]
            )
            if policy_n == self.COOP_POLICY_IDX:
                samples_copy = filter_sample_batch(
                    samples_copy, remove=True, filter_key=PUNISHING
                )
            elif policy_n == self.COOP_OPP_POLICY_IDX:
                samples_copy[samples_copy.ACTIONS] = np.array(
                    samples_copy.data[postprocessing.OPPONENT_ACTIONS]
                )
                samples_copy = filter_sample_batch(
                    samples_copy, remove=True, filter_key=BEING_PUNISHED
                )
            else:
                raise ValueError()
        elif policy_n == self.PUNITIVE_POLICY_IDX:
            samples_copy[samples_copy.REWARDS] = np.array(
                samples_copy.data[postprocessing.OPPONENT_NEGATIVE_REWARD]
            )
        elif policy_n == self.SPL_OPP_POLICY_IDX:
            samples_copy[samples_copy.ACTIONS] = np.array(
                samples_copy.data[postprocessing.OPPONENT_ACTIONS]
            )
            # SPL doesn't learn on the steps where the opponent is punishing
            # this policy. This is because the opponent is not using its
            # cooperative policy at that time.
            samples_copy = filter_sample_batch(
                samples_copy, remove=True, filter_key=BEING_PUNISHED
            )
        else:
            raise ValueError()

        return samples_copy

    def on_episode_step(self, episode, policy_id, policy_ids, *args, **kwargs):
        (
            opp_previous_obs,
            opp_a,
            being_punished_by_opp,
        ) = self._get_information_from_opponent(episode, policy_id, policy_ids)

        self.being_punished_by_opp = being_punished_by_opp
        if not self.being_punished_by_opp:
            self.n_steps_since_start += 1
            self._put_log_likelihood_in_data_buffer(
                opp_previous_obs, opp_a, self.data_queue
            )

        if self.remaining_punishing_time > 0:
            self.n_punishment_steps_in_current_epi += 1
        else:
            self.n_cooperation_steps_in_current_epi += 1

        self.observed_n_step_in_current_epi += 1

    def _get_information_from_opponent(self, episode, agent_id, agent_ids):
        opp_agent_id = [one_id for one_id in agent_ids if one_id != agent_id][
            0
        ]
        opp_a = episode.last_action_for(opp_agent_id)
        being_punished_by_opp = episode.last_pi_info_for(opp_agent_id).get(
            PUNISHING, False
        )
        return self.opp_previous_obs, opp_a, being_punished_by_opp

    def on_observation_fn(self, opp_new_obs):
        # Episode provide the last action with the given last
        # observation produced by this action. But we need the
        # observation that cause the agent to play this action
        # thus the observation n-1
        self.opp_previous_obs = self.opp_new_obs
        self.opp_new_obs = opp_new_obs

    def on_episode_end(self, base_env, *args, **kwargs):
        assert (
            self.observed_n_step_in_current_epi
            == base_env.get_unwrapped()[0].max_steps
        ), (
            "Each epi, LTFT must observe the opponent each step. "
            f"Observed {self.observed_n_step_in_current_epi} times for "
            f"{base_env.get_unwrapped()[0].max_steps} steps per episodes."
        )
        self.observed_n_step_in_current_epi = 0

        if (
            self.n_steps_since_start
            >= self.length_of_history + self.WARMUP_LENGTH
        ):
            delta_log_likelihood_coop_at_chosen_percentile = (
                self._compare_log_likelihood_on_boostrapped_sequences(
                    self.data_queue
                )
            )
            self._update_defection_metric(
                epi_defection_metric=-delta_log_likelihood_coop_at_chosen_percentile
            )

        if self.remaining_punishing_time <= -(
            self.min_coop_epi_after_punishing - 1
        ):
            if self.defection_metric > self.defection_threshold:
                self.detected_defection = True

        # Averaged by episode
        self._to_log["coop_frac"] = self.n_cooperation_steps_in_current_epi / (
            self.n_punishment_steps_in_current_epi
            + self.n_cooperation_steps_in_current_epi
        )
        self.n_cooperation_steps_in_current_epi = 0
        self.n_punishment_steps_in_current_epi = 0

        if self.remaining_punishing_time > -self.min_coop_epi_after_punishing:
            self.remaining_punishing_time -= 1

        # Switch from coop to punishment only at the start of epi
        if self.detected_defection and not self.being_punished_by_opp:
            self.active_algo_idx = self.PUNITIVE_POLICY_IDX
            self.remaining_punishing_time = self.punishment_time
            print("DEFECTION DETECTED")
            logger.info("DEFECTION DETECTED")

        if self.remaining_punishing_time <= 0:
            self.active_algo_idx = self.COOP_POLICY_IDX

        self._to_log[
            "remaining_punishing_time"
        ] = self.remaining_punishing_time
        self._to_log["active_algo_idx"] = self.active_algo_idx
        self._to_log["detected_defection"] = self.detected_defection
        self._to_log["being_punished_by_LE"] = self.being_punished_by_opp
        self.detected_defection = False

    @override(Policy)
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple["Policy", SampleBatch]]
        ] = None,
        episode: Optional["MultiAgentEpisode"] = None,
    ) -> SampleBatch:
        return self.add_welfare_fn(
            self, sample_batch, other_agent_batches, episode
        )

    def _put_log_likelihood_in_data_buffer(self, s, a, data_queue, log=True):
        s = torch.from_numpy(s).unsqueeze(dim=0)
        log_likelihood_opponent_cooperating = compute_log_likelihoods(
            self.algorithms[self.COOP_OPP_POLICY_IDX], [a], s
        )
        log_likelihood_approximated_opponent = compute_log_likelihoods(
            self.algorithms[self.SPL_OPP_POLICY_IDX], [a], s
        )
        log_likelihood_opponent_cooperating = float(
            log_likelihood_opponent_cooperating
        )
        log_likelihood_approximated_opponent = float(
            log_likelihood_approximated_opponent
        )

        self._to_log[
            "log_likelihood_opponent_cooperating"
        ] = log_likelihood_opponent_cooperating
        self._to_log[
            "log_likelihood_approximated_opponent"
        ] = log_likelihood_approximated_opponent
        data_queue.append(
            [
                log_likelihood_opponent_cooperating,
                log_likelihood_approximated_opponent,
            ]
        )

    def _bootstrap_replicats(self, data_queue, last_step_is_mandatory):
        data_array = np.array(list(data_queue), dtype=np.object)
        maximum_idx = data_array.shape[0] - 1
        bstrap_idx = np.random.randint(
            0,
            high=maximum_idx + 1,
            size=(
                self.n_bootstrap_replicates,
                self.n_steps_in_bootstrap_replicates,
            ),
        )
        if last_step_is_mandatory:
            # TODO only add it if it is not already present
            bstrap_idx[:, -1] = [
                maximum_idx for _ in range(self.n_bootstrap_replicates)
            ]
        bstrap_replts_data = data_array[bstrap_idx]
        return bstrap_replts_data

    def _compare_log_likelihood_on_boostrapped_sequences(
        self, data_queue, log=True, last_step_is_mandatory=False
    ):

        bstrap_replts_data = self._bootstrap_replicats(
            data_queue, last_step_is_mandatory
        )

        # Sum log_likelihood over u steps
        log_lik_cooperate = bstrap_replts_data[:, :, 0].sum(axis=1)

        log_lik_defect = bstrap_replts_data[:, :, 1].sum(axis=1)

        # Defect if in more than 0.95 of the replicates,
        # the actual policy is more likely than the simulated coop policy
        delta_log_likelihood_coop = log_lik_cooperate - log_lik_defect
        assert len(delta_log_likelihood_coop) == self.n_bootstrap_replicates
        delta_log_likelihood_coop_at_chosen_percentile = np.percentile(
            delta_log_likelihood_coop,
            self.percentile_for_likelihood_test,
            interpolation="linear",
        )
        delta_log_likelihood_coop_at_percentile_50 = np.percentile(
            delta_log_likelihood_coop, 50, interpolation="linear"
        )

        if log:
            mean_log_lik_cooperate = log_lik_cooperate.mean()
            mean_log_lik_defect = log_lik_defect.mean()
            self._to_log.update(
                {
                    "mean_log_lik_cooperate": mean_log_lik_cooperate,
                    "mean_log_lik_defect": mean_log_lik_defect,
                    "delta_log_likelihood_coop_at_chosen_percentile": delta_log_likelihood_coop_at_chosen_percentile,
                    "delta_log_likelihood_coop_at_percentile_50": delta_log_likelihood_coop_at_percentile_50,
                    "delta_log_likelihood_coop_std": delta_log_likelihood_coop.std(),
                    "delta_log_likelihood_coop_mean": delta_log_likelihood_coop.mean(),
                }
            )
        return delta_log_likelihood_coop_at_chosen_percentile

    def _update_defection_metric(self, epi_defection_metric):
        self.defection_carac_queue.append(epi_defection_metric)
        self.defection_metric = sum(self.defection_carac_queue) / len(
            self.defection_carac_queue
        )
        self._to_log["defection_metric"] = round(
            float(self.defection_metric), 4
        )

    def on_postprocess_trajectory(
        self,
        policy_id: PolicyID,
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, SampleBatch],
        *args,
        **kwargs,
    ):

        all_agent_keys = list(original_batches.keys())
        all_agent_keys.remove(policy_id)
        assert len(all_agent_keys) == 1, "LTFT only works for 2 agents"
        opp_policy_id = all_agent_keys[0]

        opp_is_broadcast_punishment_state = (
            PUNISHING in original_batches[opp_policy_id][1].keys()
        )
        if opp_is_broadcast_punishment_state:
            postprocessed_batch.data[BEING_PUNISHED] = copy.deepcopy(
                original_batches[all_agent_keys[0]][1][PUNISHING]
            )
        else:
            postprocessed_batch.data[BEING_PUNISHED] = [False] * len(
                postprocessed_batch[postprocessed_batch.OBS]
            )


# Modified from torch_policy
def compute_log_likelihoods(
    policy,
    actions: Union[List[TensorType], TensorType],
    obs_batch: Union[List[TensorType], TensorType],
    state_batches: Optional[List[TensorType]] = None,
    prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
    prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
) -> TensorType:

    if policy.action_sampler_fn and policy.action_distribution_fn is None:
        raise ValueError(
            "Cannot compute log-prob/likelihood w/o an "
            "`action_distribution_fn` and a provided "
            "`action_sampler_fn`!"
        )

    with torch.no_grad():
        input_dict = policy._lazy_tensor_dict(
            {SampleBatch.CUR_OBS: obs_batch, SampleBatch.ACTIONS: actions}
        )
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
        state_batches = [
            convert_to_torch_tensor(s, policy.device)
            for s in (state_batches or [])
        ]

        # Exploration hook before each forward pass.
        policy.exploration.before_compute_actions(explore=False)

        # Action dist class and inputs are generated via custom function.
        if policy.action_distribution_fn:

            # Try new action_distribution_fn signature, supporting
            # state_batches and seq_lens.
            try:
                (
                    dist_inputs,
                    dist_class,
                    state_out,
                ) = policy.action_distribution_fn(
                    policy,
                    policy.model,
                    input_dict=input_dict,
                    state_batches=state_batches,
                    seq_lens=seq_lens,
                    explore=False,
                    is_training=False,
                )
            # Trying the old way (to stay backward compatible).
            # TODO: Remove in future.
            except TypeError as e:
                if (
                    "positional argument" in e.args[0]
                    or "unexpected keyword argument" in e.args[0]
                ):
                    dist_inputs, dist_class, _ = policy.action_distribution_fn(
                        policy=policy,
                        model=policy.model,
                        obs_batch=input_dict[SampleBatch.CUR_OBS],
                        explore=False,
                        is_training=False,
                    )
                else:
                    raise e

        # Default action-dist inputs calculation.
        else:
            dist_class = policy.dist_class
            dist_inputs, _ = policy.model(input_dict, state_batches, seq_lens)

        action_dist = dist_class(dist_inputs, policy.model)
        # ADDITION 1/1 STARTS
        if policy.config["explore"]:
            # Adding that because of a bug in TorchCategorical
            #  which modify dist_inputs through action_dist:
            _ = policy.exploration.get_exploration_action(
                action_distribution=action_dist,
                timestep=policy.global_timestep,
                explore=policy.config["explore"],
            )
            action_dist = dist_class(dist_inputs, policy.model)
        # ADDITION 1/1 ENDS

        log_likelihoods = action_dist.logp(input_dict[SampleBatch.ACTIONS])

        return log_likelihoods


LTFTCallbacks = callbacks.PolicyCallbacks


def observation_fn(
    agent_obs,
    worker: "RolloutWorker",
    base_env: BaseEnv,
    policies: Dict[PolicyID, Policy],
    episode: MultiAgentEpisode,
):
    agent_ids = list(policies.keys())
    assert len(agent_ids) == 2, "LTFT Implemented for 2 players"

    for agent_id, policy in policies.items():
        if isinstance(policy, LTFTTorchPolicy):
            opp_agent_id = [
                one_id for one_id in agent_ids if one_id != agent_id
            ][0]
            opp_raw_obs = agent_obs[opp_agent_id]
            filtered_obs = postprocessing.apply_preprocessors(
                worker, opp_raw_obs, opp_agent_id
            )

            policy.on_observation_fn(copy.deepcopy(filtered_obs))

    return agent_obs
