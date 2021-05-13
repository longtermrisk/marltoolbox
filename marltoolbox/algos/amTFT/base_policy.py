import copy
import logging
from typing import Dict, TYPE_CHECKING

import numpy as np
import torch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import override
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_ops import (
    convert_to_torch_tensor,
)
from ray.rllib.utils.typing import PolicyID

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

from marltoolbox.algos import hierarchical
from marltoolbox.algos.amTFT.weights_exchanger import WeightsExchanger
from marltoolbox.utils import postprocessing, restore, callbacks
from marltoolbox.algos.amTFT.base import (
    AmTFTReferenceClass,
    WORKING_STATES,
    WORKING_STATES_IN_EVALUATION,
    OPP_COOP_POLICY_IDX,
    OWN_COOP_POLICY_IDX,
    OWN_SELFISH_POLICY_IDX,
)

logger = logging.getLogger(__name__)


class AmTFTPolicyBase(
    hierarchical.HierarchicalTorchPolicy, WeightsExchanger, AmTFTReferenceClass
):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)

        self.total_debit = 0
        self.n_steps_to_punish = 0
        self.observed_n_step_in_current_epi = 0
        self.last_own_algo_idx_in_eval = OWN_COOP_POLICY_IDX
        self.opp_previous_obs = None
        self.opp_new_obs = None
        self.own_previous_obs = None
        self.own_new_obs = None
        self.both_previous_raw_obs = None
        self.both_new_raw_obs = None
        self.coop_own_rnn_state_before_last_act = None
        self.coop_opp_rnn_state_before_last_act = None
        # notation T in the paper
        self.debit_threshold = config["debit_threshold"]
        # notation alpha in the paper
        self.punishment_multiplier = config["punishment_multiplier"]
        self.working_state = config["working_state"]
        assert (
            self.working_state in WORKING_STATES
        ), f"self.working_state {self.working_state}"
        self.verbose = config["verbose"]
        self.welfare_key = config["welfare_key"]
        self.auto_load_checkpoint = config.get("auto_load_checkpoint", True)
        self.punish_instead_of_selfish = config.get(
            "punish_instead_of_selfish", False
        )
        self.punish_instead_of_selfish_key = (
            postprocessing.OPPONENT_NEGATIVE_REWARD
        )

        if self.working_state in WORKING_STATES_IN_EVALUATION:
            self._set_models_for_evaluation()

        if (
            self.auto_load_checkpoint
            and restore.LOAD_FROM_CONFIG_KEY in config.keys()
        ):
            restore.before_loss_init_load_policy_checkpoint(self)

    def _set_models_for_evaluation(self):
        for algo in self.algorithms:
            algo.model.eval()

    @with_lock
    @override(hierarchical.HierarchicalTorchPolicy)
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        state_batches = self._select_witch_algo_to_use(state_batches)

        self._track_last_coop_rnn_state(state_batches)
        actions, state_out, extra_fetches = super()._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )
        if self.verbose > 1:
            print("algo idx", self.active_algo_idx, "action", actions)
            print("extra_fetches", extra_fetches)
            print("state_batches (in)", state_batches)
            print("state_out", state_out)

        return actions, state_out, extra_fetches

    def _select_witch_algo_to_use(self, state_batches):
        if (
            self.working_state == WORKING_STATES[0]
            or self.working_state == WORKING_STATES[4]
        ):
            self.active_algo_idx = OWN_COOP_POLICY_IDX
        elif (
            self.working_state == WORKING_STATES[1]
            or self.working_state == WORKING_STATES[3]
        ):
            self.active_algo_idx = OWN_SELFISH_POLICY_IDX

        elif self.working_state == WORKING_STATES[2]:
            state_batches = self._select_algo_to_use_in_eval(state_batches)
        else:
            raise ValueError(
                f'config["working_state"] ' f"must be one of {WORKING_STATES}"
            )

        return state_batches

    def _select_algo_to_use_in_eval(self, state_batches):
        if self.n_steps_to_punish == 0:
            self.active_algo_idx = OWN_COOP_POLICY_IDX
        elif self.n_steps_to_punish > 0:
            self.active_algo_idx = OWN_SELFISH_POLICY_IDX
            self.n_steps_to_punish -= 1
        else:
            raise ValueError("self.n_steps_to_punish can't be below zero")

        state_batches = self._check_for_rnn_state_reset(
            state_batches, "last_own_algo_idx_in_eval"
        )

        return state_batches

    def _check_for_rnn_state_reset(self, state_batches, last_algo_idx: str):
        if getattr(self, last_algo_idx) != self.active_algo_idx:
            state_batches = self._get_initial_rnn_state(state_batches)
            self._to_log["reset_rnn_state"] = self.active_algo_idx
            setattr(self, last_algo_idx, self.active_algo_idx)
            if self.verbose > 0:
                print("reset_rnn_state")
        else:
            if "reset_rnn_state" in self._to_log.keys():
                self._to_log.pop("reset_rnn_state")
        return state_batches

    def _get_initial_rnn_state(self, state_batches):
        if "model" in self.config.keys() and self.config["model"]["use_lstm"]:
            initial_state = self.algorithms[
                self.active_algo_idx
            ].get_initial_state()
            initial_state = [
                convert_to_torch_tensor(s, self.device) for s in initial_state
            ]
            initial_state = [s.unsqueeze(0) for s in initial_state]
            msg = (
                f"self.active_algo_idx {self.active_algo_idx} "
                f"state_batches {state_batches} reset to initial rnn state"
            )
            # print(msg)
            logger.info(msg)
            return initial_state
        else:
            return state_batches

    def _track_last_coop_rnn_state(self, state_batches):
        if self.active_algo_idx == OWN_COOP_POLICY_IDX:
            self.coop_own_rnn_state_before_last_act = state_batches

    @override(hierarchical.HierarchicalTorchPolicy)
    def _learn_on_batch(self, samples: SampleBatch):

        if self.working_state == WORKING_STATES[0]:
            algo_idx_to_train = OWN_COOP_POLICY_IDX
        elif self.working_state == WORKING_STATES[1]:
            algo_idx_to_train = OWN_SELFISH_POLICY_IDX
        else:
            raise ValueError(
                f"self.working_state must be one of " f"{WORKING_STATES[0:2]}"
            )
        samples = self._modify_batch_for_policy(algo_idx_to_train, samples)

        algo_to_train = self.algorithms[algo_idx_to_train]
        learner_stats = {"learner_stats": {}}
        learner_stats["learner_stats"][
            f"algo{algo_idx_to_train}"
        ] = algo_to_train.learn_on_batch(samples)

        if self.verbose > 1:
            print(f"learn_on_batch WORKING_STATES " f"{self.working_state}")

        return learner_stats

    def _modify_batch_for_policy(self, algo_idx_to_train, samples):
        if algo_idx_to_train == OWN_COOP_POLICY_IDX:
            samples = samples.copy()
            samples = self._overwrite_reward_for_policy_in_use(
                samples, self.welfare_key
            )
        elif (
            self.punish_instead_of_selfish
            and algo_idx_to_train == OWN_SELFISH_POLICY_IDX
        ):
            samples = samples.copy()
            samples = self._overwrite_reward_for_policy_in_use(
                samples, self.punish_instead_of_selfish_key
            )

        return samples

    def _overwrite_reward_for_policy_in_use(self, samples_copy, welfare_key):
        samples_copy[samples_copy.REWARDS] = np.array(
            samples_copy.data[welfare_key]
        )
        logger.debug(f"overwrite reward with {welfare_key}")
        return samples_copy

    def on_observation_fn(self, own_new_obs, opp_new_obs, both_new_raw_obs):
        # Episode provide the last action with the given last
        # observation produced by this action. But we need the
        # observation that cause the agent to play this action
        # thus the observation n-1
        if self.own_new_obs is not None:
            self.own_previous_obs = self.own_new_obs
            self.opp_previous_obs = self.opp_new_obs
            self.both_previous_raw_obs = self.both_new_raw_obs
        self.own_new_obs = own_new_obs
        self.opp_new_obs = opp_new_obs
        self.both_new_raw_obs = both_new_raw_obs

    def on_episode_step(
        self,
        policy_id,
        policy,
        policy_ids,
        episode,
        worker,
        base_env,
        env_index,
        *args,
        **kwargs,
    ):
        opp_obs, raw_obs, opp_a = self._get_information_from_opponent(
            policy_id, policy_ids, episode
        )

        # Ignored the first step in epi because the
        # actions provided are fake (they were not played)
        self._on_episode_step(
            opp_obs, raw_obs, opp_a, worker, base_env, episode, env_index
        )

        self.observed_n_step_in_current_epi += 1

    def _get_information_from_opponent(self, agent_id, agent_ids, episode):
        opp_agent_id = [one_id for one_id in agent_ids if one_id != agent_id][
            0
        ]
        opp_a = episode.last_action_for(opp_agent_id)

        return self.opp_previous_obs, self.both_previous_raw_obs, opp_a

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
        if self.working_state == WORKING_STATES[2]:

            if self._is_punishment_planned():
                self._to_log["punishing"] = True
                # self.n_steps_to_punish -= 1 Must not be here to allow
                # to use n_steps_to_punish during rollouts
                #  during which on_episode_step is not called
                if self.verbose > 0:
                    print(
                        f"punishing self.n_steps_to_punish: "
                        f"{self.n_steps_to_punish}"
                    )
            else:
                self._to_log["punishing"] = False
                coop_opp_simulated_action = (
                    self._simulate_action_from_cooperative_opponent(opp_obs)
                )
                assert (
                    len(worker.async_env.env_states) == 1
                ), "amTFT in eval only works with one env not vector of envs"
                assert (
                    worker.env.step_count_in_current_episode
                    == worker.async_env.env_states[
                        0
                    ].env.step_count_in_current_episode
                )
                assert (
                    worker.env.step_count_in_current_episode
                    == self._base_env_at_last_step.get_unwrapped()[
                        0
                    ].step_count_in_current_episode
                    + 1
                )
                self._update_total_debit(
                    last_obs,
                    opp_action,
                    worker,
                    self._base_env_at_last_step,
                    episode,
                    env_index,
                    coop_opp_simulated_action,
                )
                if self._is_starting_new_punishment_required():
                    self._plan_punishment(
                        opp_action, coop_opp_simulated_action, worker, last_obs
                    )

            self._base_env_at_last_step = copy.deepcopy(base_env)

            self._to_log["n_steps_to_punish"] = self.n_steps_to_punish
            self._to_log["debit_threshold"] = self.debit_threshold

    def _is_punishment_planned(self):
        return self.n_steps_to_punish > 0

    def on_episode_start(self, *args, **kwargs):
        if self.working_state in WORKING_STATES_IN_EVALUATION:
            self._base_env_at_last_step = copy.deepcopy(kwargs["base_env"])
            self.last_own_algo_idx_in_eval = OWN_COOP_POLICY_IDX
            self.coop_opp_rnn_state_after_last_act = self.algorithms[
                OPP_COOP_POLICY_IDX
            ].get_initial_state()

    def _simulate_action_from_cooperative_opponent(self, opp_obs):
        if self.verbose > 1:
            print("opp_obs for opp coop simu nonzero obs", np.nonzero(opp_obs))
            for i, algo in enumerate(self.algorithms):
                print("algo", i, algo)
        self.coop_opp_rnn_state_before_last_act = (
            self.coop_opp_rnn_state_after_last_act
        )
        (
            coop_opp_simulated_action,
            self.coop_opp_rnn_state_after_last_act,
            coop_opp_extra_fetches,
        ) = self.algorithms[OPP_COOP_POLICY_IDX].compute_single_action(
            obs=opp_obs,
            state=self.coop_opp_rnn_state_after_last_act,
        )
        if self.verbose > 1:
            print(
                coop_opp_simulated_action,
                "coop_opp_extra_fetches",
                coop_opp_extra_fetches,
            )
            print(
                "state before simu coop opp",
                self.coop_opp_rnn_state_before_last_act,
            )
            print(
                "state after simu coop opp",
                self.coop_opp_rnn_state_after_last_act,
            )

        return coop_opp_simulated_action

    def _update_total_debit(
        self,
        last_obs,
        opp_action,
        worker,
        base_env,
        episode,
        env_index,
        coop_opp_simulated_action,
    ):
        if self.verbose > 1:
            print(
                self.own_policy_id,
                self.config[restore.LOAD_FROM_CONFIG_KEY][0].split("/")[-5],
            )
        if coop_opp_simulated_action != opp_action:
            if self.verbose > 0:
                print(
                    self.own_policy_id,
                    "coop_opp_simulated_action != opp_action:",
                    coop_opp_simulated_action,
                    opp_action,
                )
            if (
                worker.env.step_count_in_current_episode
                >= worker.env.max_steps
            ):
                debit = 0
            else:
                debit = self._compute_debit(
                    last_obs,
                    opp_action,
                    worker,
                    base_env,
                    episode,
                    env_index,
                    coop_opp_simulated_action,
                )
        else:
            if self.verbose > 0:
                print(
                    "id",
                    self.own_policy_id,
                    "coop_opp_simulated_action == opp_action",
                )
            debit = 0
        tmp = self.total_debit
        self.total_debit += debit
        self._to_log["debit_this_step"] = debit
        self._to_log["total_debit"] = self.total_debit
        self._to_log["summed_debit"] = (
            debit + self._to_log["summed_debit"]
            if "summed_debit" in self._to_log
            else debit
        )
        if coop_opp_simulated_action != opp_action:
            if self.verbose > 0:
                print(f"debit {debit}")
                print(
                    f"self.total_debit {self.total_debit}, previous was {tmp}"
                )

    def _is_starting_new_punishment_required(self, manual_threshold=None):
        if manual_threshold is not None:
            return self.total_debit >= manual_threshold
        return self.total_debit >= self.debit_threshold

    def _plan_punishment(
        self, opp_action, coop_opp_simulated_action, worker, last_obs
    ):
        if worker.env.step_count_in_current_episode >= worker.env.max_steps:
            self.n_steps_to_punish = 0
        else:
            self.n_steps_to_punish = self._compute_punishment_duration(
                opp_action, coop_opp_simulated_action, worker, last_obs
            )

        self.total_debit = 0
        self._to_log["n_steps_to_punish"] = self.n_steps_to_punish
        self._to_log["summed_n_steps_to_punish"] = (
            self.n_steps_to_punish + self._to_log["summed_n_steps_to_punish"]
            if "summed_n_steps_to_punish" in self._to_log
            else self.n_steps_to_punish
        )
        if self.verbose > 0:
            print(f"reset self.total_debit to 0 since planned punishement")

    def on_episode_end(self, *args, **kwargs):
        self._defensive_check_observed_n_opp_moves(*args, **kwargs)
        self._if_in_eval_reset_debit_and_punish()

    def _defensive_check_observed_n_opp_moves(self, *args, **kwargs):
        if self.working_state in WORKING_STATES_IN_EVALUATION:
            assert (
                self.observed_n_step_in_current_epi
                == kwargs["base_env"].get_unwrapped()[0].max_steps
            ), (
                "Each epi, LTFT must observe the opponent each step. "
                f"Observed {self.observed_n_step_in_current_epi} times for "
                f"{kwargs['base_env'].get_unwrapped()[0].max_steps} "
                "steps per episodes."
            )
        self.observed_n_step_in_current_epi = 0

    def _if_in_eval_reset_debit_and_punish(self):
        if self.working_state in WORKING_STATES_IN_EVALUATION:
            self.total_debit = 0
            self.n_steps_to_punish = 0
            if self.verbose > 0:
                logger.info(
                    "reset self.total_debit to 0 since end of " "episode"
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
        raise NotImplementedError()

    def _compute_punishment_duration(
        self, opp_action, coop_opp_simulated_action, worker, last_obs
    ):
        raise NotImplementedError()

    def _get_last_rnn_states_before_rollouts(self):
        if self.config["model"]["use_lstm"]:
            return {
                self.own_policy_id: self._squeezes_rnn_state(
                    self.coop_own_rnn_state_before_last_act
                ),
                self.opp_policy_id: self.coop_opp_rnn_state_before_last_act,
            }

        else:
            return None

    @staticmethod
    def _squeezes_rnn_state(state):
        return [
            s.squeeze(0)
            if torch and isinstance(s, torch.Tensor)
            else np.squeeze(s, 0)
            for s in state
        ]


class AmTFTCallbacks(callbacks.PolicyCallbacks):
    def on_train_result(self, trainer, *args, **kwargs):
        """
        We only call this method one time for both policies in training.
        """
        local_policy_map = trainer.workers.local_worker().policy_map
        assert len(local_policy_map) == 2
        one_policy_id = list(local_policy_map.keys())[0]

        self._call_method_from_policy(
            *args,
            worker=trainer.workers.local_worker(),
            method="on_train_result",
            policy=local_policy_map[one_policy_id],
            policy_id=one_policy_id,
            trainer=trainer,
            **kwargs,
        )


def observation_fn(
    agent_obs,
    worker: "RolloutWorker",
    base_env: BaseEnv,
    policies: Dict[PolicyID, Policy],
    episode: MultiAgentEpisode,
):
    agent_ids = list(policies.keys())
    assert len(agent_ids) == 2, "amTFT Implemented for 2 players"

    for agent_id, policy in policies.items():
        if isinstance(policy, AmTFTReferenceClass):
            opp_agent_id = [
                one_id for one_id in agent_ids if one_id != agent_id
            ][0]
            both_raw_obs = agent_obs
            own_raw_obs = agent_obs[agent_id]
            filtered_own_obs = postprocessing.apply_preprocessors(
                worker, own_raw_obs, agent_id
            )
            opp_raw_obs = agent_obs[opp_agent_id]
            filtered_opp_obs = postprocessing.apply_preprocessors(
                worker, opp_raw_obs, opp_agent_id
            )

            policy.on_observation_fn(
                own_new_obs=copy.deepcopy(filtered_own_obs),
                opp_new_obs=copy.deepcopy(filtered_opp_obs),
                both_new_raw_obs=copy.deepcopy(both_raw_obs),
            )

    return agent_obs
