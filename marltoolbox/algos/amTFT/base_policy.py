import copy
import logging
from typing import List, Union, Optional, Dict, Tuple, TYPE_CHECKING

from ray.rllib.agents.dqn.dqn_torch_policy import postprocess_nstep_and_prio
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts, override
from ray.rllib.utils.typing import TensorType, PolicyID

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

from marltoolbox.algos import hierarchical, augmented_dqn
from marltoolbox.algos.amTFT.weights_exchanger import \
    WeightsExchangerWtExploiter
from marltoolbox.utils import \
    postprocessing, miscellaneous, restore, callbacks

logger = logging.getLogger(__name__)

APPROXIMATION_METHOD_Q_VALUE = "amTFT_use_Q_net"
APPROXIMATION_METHOD_ROLLOUTS = "amTFT_use_rollout"
APPROXIMATION_METHODS = (APPROXIMATION_METHOD_Q_VALUE,
                         APPROXIMATION_METHOD_ROLLOUTS)
WORKING_STATES = ("train_coop",
                  "train_selfish",
                  "eval_amtft",
                  "eval_naive_selfish",
                  "eval_naive_coop")
WORKING_STATES_IN_EVALUATION = WORKING_STATES[2:]

OWN_COOP_POLICY_IDX = 0
OWN_SELFISH_POLICY_IDX = 1
OPP_COOP_POLICY_IDX = 2
OPP_SELFISH_POLICY_IDX = 3

DEFAULT_NESTED_POLICY_SELFISH = augmented_dqn.MyDQNTorchPolicy
DEFAULT_NESTED_POLICY_COOP = DEFAULT_NESTED_POLICY_SELFISH.with_updates(
    postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
        postprocessing.welfares_postprocessing_fn(
            add_utilitarian_welfare=True, ),
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
        "rollout_length": 40,
        "n_rollout_replicas": 20,
        "last_k": 1,
        # TODO use log level of RLLib instead of mine
        "verbose": 1,

        # To configure
        "own_policy_id": None,
        "opp_policy_id": None,
        "callbacks": None,
        # One from marltoolbox.utils.postprocessing.WELFARES
        "welfare": postprocessing.WELFARE_UTILITARIAN,

        'nested_policies': [
            # Here the trainer need to be a DQNTrainer
            # to provide the config for the 3 DQNTorchPolicy
            {"Policy_class": DEFAULT_NESTED_POLICY_COOP,
             "config_update": {}},
            {"Policy_class": DEFAULT_NESTED_POLICY_SELFISH,
             "config_update": {}},
            {"Policy_class": DEFAULT_NESTED_POLICY_COOP,
             "config_update": {}},
            {"Policy_class": DEFAULT_NESTED_POLICY_SELFISH,
             "config_update": {}},
        ],
        "optimizer": {"sgd_momentum": 0.0, },
    }
)

PLOT_KEYS = [
    "punish",
    "debit",
    "debit_threshold",
    "summed_debit",
    "summed_n_steps_to_punish"
]

PLOT_ASSEMBLAGE_TAGS = [
    ("punish",),
    ("debit",),
    ("debit_threshold",),
    ("summed_debit",),
    ("summed_n_steps_to_punish",),
]


class AmTFTReferenceClass:
    pass


class AmTFTPolicyBase(hierarchical.HierarchicalTorchPolicy,
                      WeightsExchangerWtExploiter,
                      AmTFTReferenceClass):

    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)

        self.total_debit = 0
        self.n_steps_to_punish = 0
        self.observed_n_step_in_current_epi = 0
        self._first_fake_step_played = False
        self.opp_previous_obs = None
        self.opp_new_obs = None
        self.both_previous_raw_obs = None
        self.both_new_raw_obs = None
        # notation T in the paper
        self.debit_threshold = config["debit_threshold"]
        # notation alpha in the paper
        self.punishment_multiplier = config["punishment_multiplier"]
        self.welfare = config["welfare"]
        self.working_state = config["working_state"]
        self.verbose = config["verbose"]

        assert self.welfare in postprocessing.WELFARES, \
            f"self.welfare: {self.welfare} must be in " \
            f"postprocessing.WELFARES: {postprocessing.WELFARES}"

        if self.working_state in WORKING_STATES_IN_EVALUATION:
            self._set_models_for_evaluation()

        restore.after_init_load_policy_checkpoint(self)

    def _set_models_for_evaluation(self):
        for algo in self.algorithms:
            algo.model.eval()

    @override(hierarchical.HierarchicalTorchPolicy)
    def compute_actions(self, obs_batch: Union[List[TensorType], TensorType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[
                            List[TensorType], TensorType] = None,
                        prev_reward_batch: Union[
                            List[TensorType], TensorType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["MultiAgentEpisode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        self._select_witch_algo_to_use()
        actions, state_out, extra_fetches = \
            self.algorithms[self.active_algo_idx].compute_actions(
                obs_batch,
                state_batches,
                prev_action_batch,
                prev_reward_batch,
                info_batch,
                episodes,
                explore,
                timestep,
                **kwargs)
        if self.verbose > 2:
            print(f"self.active_algo_idx {self.active_algo_idx}")
        return actions, state_out, extra_fetches

    def _select_witch_algo_to_use(self):
        if self.working_state == WORKING_STATES[0] or \
                self.working_state == WORKING_STATES[4]:
            self.active_algo_idx = OWN_COOP_POLICY_IDX
        elif self.working_state == WORKING_STATES[1] or \
                self.working_state == WORKING_STATES[3]:
            self.active_algo_idx = OWN_SELFISH_POLICY_IDX
        elif self.working_state == WORKING_STATES[2]:
            self._select_algo_to_use_in_eval()
        else:
            raise ValueError(f'config["working_state"] '
                             f'must be one of {WORKING_STATES}')

    def _select_algo_to_use_in_eval(self):
        if self.n_steps_to_punish == 0:
            self.active_algo_idx = OWN_COOP_POLICY_IDX
        elif self.n_steps_to_punish > 0:
            self.active_algo_idx = OWN_SELFISH_POLICY_IDX
            self.n_steps_to_punish -= 1
        else:
            raise ValueError("self.n_steps_to_punish can't be below zero")

    @override(hierarchical.HierarchicalTorchPolicy)
    def learn_on_batch(self, samples: SampleBatch):

        working_state_idx = WORKING_STATES.index(self.working_state)
        assert working_state_idx == OWN_COOP_POLICY_IDX \
               or working_state_idx == OWN_SELFISH_POLICY_IDX, \
            f"current working_state is {self.working_state} " \
            f"but you be one of " \
            f"[{WORKING_STATES[OWN_COOP_POLICY_IDX]}, " \
            f"{WORKING_STATES[OWN_SELFISH_POLICY_IDX]}]"

        self._update_lr_in_all_optimizers()

        algo_to_train = self.algorithms[working_state_idx]
        learner_stats = {"learner_stats": {}}
        learner_stats["learner_stats"][f"algo{working_state_idx}"] = \
            algo_to_train.learn_on_batch(samples)

        if self.verbose > 1:
            print(f"learn_on_batch WORKING_STATES "
                  f"{WORKING_STATES[working_state_idx]}, ")

        self._log_learning_rates()

        return learner_stats

    def _update_lr_in_all_optimizers(self):
        self.optimizer()

    def on_observation_fn(self, opp_new_obs, both_new_raw_obs):
        # Episode provide the last action with the given last
        # observation produced by this action. But we need the
        # observation that cause the agent to play this action
        # thus the observation n-1
        if self._first_fake_step_played:
            self.opp_previous_obs = self.opp_new_obs
            self.both_previous_raw_obs = self.both_new_raw_obs
        self.opp_new_obs = opp_new_obs
        self.both_new_raw_obs = both_new_raw_obs

    def on_episode_step(
            self, policy_id, policy, policy_ids, episode, worker,
            base_env, env_index, *args, **kwargs):
        if self._first_fake_step_played:
            opp_obs, raw_obs, opp_a = \
                self._get_information_from_opponent(
                    policy_id, policy_ids, episode)

            # Ignored the first step in epi because the
            # actions provided are fake (they were not played)
            policy._on_episode_step(opp_obs, raw_obs, opp_a, worker,
                                    base_env, episode, env_index)

            self.observed_n_step_in_current_epi += 1
        else:
            self._first_fake_step_played = True

    def _get_information_from_opponent(self, agent_id, agent_ids, episode):
        opp_agent_id = [one_id
                        for one_id in agent_ids
                        if one_id != agent_id][0]
        opp_a = episode.last_action_for(opp_agent_id)

        return self.opp_previous_obs, self.both_previous_raw_obs, opp_a

    def _on_episode_step(self, opp_obs, last_obs, opp_action, worker,
                         base_env, episode, env_index):
        if self.working_state == WORKING_STATES[2]:

            if self._is_punishment_planned():
                # self.n_steps_to_punish -= 1 Must not be here to allow
                # to use n_steps_to_punish during rollouts
                #  during which on_episode_step is not called
                if self.verbose > 0:
                    print(f"punishing self.n_steps_to_punish: "
                          f"{self.n_steps_to_punish}")
            else:
                coop_opp_simulated_action = \
                    self._simulate_action_from_cooperative_opponent(opp_obs)
                self._update_total_debit(
                    last_obs, opp_action, worker, base_env,
                    episode, env_index, coop_opp_simulated_action)
                if self._is_starting_new_punishment_required():
                    self._plan_punishment(
                        opp_action, coop_opp_simulated_action,
                        worker, last_obs)

            self._to_log['debit_threshold'] = self.debit_threshold

    def _is_punishment_planned(self):
        return self.n_steps_to_punish > 0

    def _simulate_action_from_cooperative_opponent(self, opp_obs):
        coop_opp_simulated_action, _, self.coop_opp_extra_fetches = \
            self.algorithms[OPP_COOP_POLICY_IDX].compute_actions([opp_obs])
        # Returned a list
        coop_opp_simulated_action = coop_opp_simulated_action[0]
        return coop_opp_simulated_action

    def _update_total_debit(self, last_obs, opp_action, worker, base_env,
                            episode, env_index, coop_opp_simulated_action):

        if coop_opp_simulated_action != opp_action:
            # if self.verbose > 0:
            #
            #
            if worker.env.step_count_in_current_episode >= \
                    worker.env.max_steps:
                debit = 0
            else:
                debit = self._compute_debit(
                    last_obs, opp_action, worker, base_env,
                    episode, env_index, coop_opp_simulated_action)
        else:
            if self.verbose > 0:
                print("coop_opp_simulated_action == opp_action")
            debit = 0
        tmp = self.total_debit
        self.total_debit += debit
        self._to_log['summed_debit'] = (
            debit + self._to_log['summed_debit']
            if 'summed_debit' in self._to_log else debit
        )
        if coop_opp_simulated_action != opp_action:
            if self.verbose > 0:
                print("coop_opp_simulated_action != opp_action:",
                      coop_opp_simulated_action, opp_action)
                print(f"debit {debit}")
                print(
                    f"self.total_debit {self.total_debit}, previous was {tmp}")

    def _is_starting_new_punishment_required(self, manual_threshold=None):
        if manual_threshold is not None:
            return self.total_debit > manual_threshold
        return self.total_debit > self.debit_threshold

    def _plan_punishment(
            self, opp_action, coop_opp_simulated_action, worker, last_obs):
        if worker.env.step_count_in_current_episode >= worker.env.max_steps:
            self.n_steps_to_punish = 0
        else:
            self.n_steps_to_punish = self._compute_punishment_duration(
                opp_action,
                coop_opp_simulated_action,
                worker,
                last_obs)

        self.total_debit = 0
        self._to_log['summed_n_steps_to_punish'] = (
            self.n_steps_to_punish + self._to_log['summed_n_steps_to_punish']
            if 'summed_n_steps_to_punish' in self._to_log else
            self.n_steps_to_punish
        )
        if self.verbose > 0:
            print(f"reset self.total_debit to 0 since planned punishement")

    def on_episode_end(self, *args, **kwargs):
        if self.working_state not in WORKING_STATES_IN_EVALUATION:
            assert self.observed_n_step_in_current_epi == \
                   kwargs["base_env"].get_unwrapped()[0].max_steps, \
                "Each epi, LTFT must observe the opponent each step. " \
                f"Observed {self.observed_n_step_in_current_epi} times for " \
                f"{kwargs['base_env'].get_unwrapped()[0].max_steps} " \
                "steps per episodes."
            self.observed_n_step_in_current_epi = 0

        if self.working_state == WORKING_STATES[2]:
            self.total_debit = 0
            self.n_steps_to_punish = 0
            if self.verbose > 0:
                print(f"reset self.total_debit to 0 since end of episode")

    def _compute_debit(self, last_obs, opp_action, worker, base_env,
                       episode, env_index, coop_opp_simulated_action):
        raise NotImplementedError()

    def _compute_punishment_duration(
            self, opp_action, coop_opp_simulated_action, worker, last_obs):
        raise NotImplementedError()


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
            **kwargs)


def observation_fn(agent_obs,
                   worker: "RolloutWorker",
                   base_env: BaseEnv,
                   policies: Dict[PolicyID, Policy],
                   episode: MultiAgentEpisode, ):
    agent_ids = list(policies.keys())
    assert len(agent_ids) == 2, "LTFT Implemented for 2 players"

    for agent_id, policy in policies.items():
        if isinstance(policy, AmTFTReferenceClass):
            opp_agent_id = [one_id
                            for one_id in agent_ids
                            if one_id != agent_id][0]
            both_raw_obs = agent_obs
            opp_raw_obs = agent_obs[opp_agent_id]
            filtered_obs = postprocessing.apply_preprocessors(
                worker, opp_raw_obs, opp_agent_id)

            policy.on_observation_fn(
                opp_new_obs=copy.deepcopy(filtered_obs),
                both_new_raw_obs=copy.deepcopy(both_raw_obs))

    return agent_obs
