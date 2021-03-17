import copy
from collections import Iterable

import logging
from typing import List, Union, Optional, Dict, Tuple

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn.dqn_torch_policy import postprocess_nstep_and_prio
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts, override
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.env import BaseEnv

from marltoolbox.algos import hierarchical, augmented_dqn
from marltoolbox.utils import postprocessing, miscellaneous, restore

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

DEFAULT_NESTED_POLICY_SELFISH = augmented_dqn.MyDQNTorchPolicy
DEFAULT_NESTED_POLICY_COOP = DEFAULT_NESTED_POLICY_SELFISH.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.welfares_postprocessing_fn(add_utilitarian_welfare=True, ),
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
        "sgd_momentum": 0.0,
    }
)


PLOT_KEYS = ["policy_reward_mean",
             "loss",
             "entropy_during_eval",
             "entropy_avg",
             "td_error",
             "punish",
             "debig",
             "debit_threshold",
             "summed_debit",
             "summed_n_steps_to_punish"

             "act_dist_inputs_avg",
             "act_dist_inputs_single",
             "q_values_avg",
             "learn_on_batch",
             "action_prob",
             "q_values_single",
             "_lr",
             "max_q_values",
             "min_q_values",
             ]

PLOT_ASSEMBLAGE_TAGS = [
    ("policy_reward_mean",),
    ("loss", "td_error"),
    ("entropy_during_eval",),
    ("entropy_avg",),

    ("punish",),
    ("debig",),
    ("debit_threshold",),
    ("summed_debit",),
    ("summed_n_steps_to_punish",),

    ("learn_on_batch",),
    ("max_q_values",),
    ("min_q_values",),
    ("act_dist_inputs_avg_act0",),
    ("act_dist_inputs_avg_act1",),
    ("act_dist_inputs_avg_act2",),
    ("act_dist_inputs_avg_act3",),
    ("q_values_avg_act0",),
    ("q_values_avg_act1",),
    ("q_values_avg_act2",),
    ("q_values_avg_act3",),
    ("q_values_single_max",),
    ("act_dist_inputs_single_max",),
    ("action_prob_single",),
    ("action_prob_avg",),
    ("_lr",),
]


class amTFTPolicyBase(hierarchical.HierarchicalTorchPolicy):

    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)

        self.total_debit = 0
        self.n_steps_to_punish = 0
        self.debit_threshold = config["debit_threshold"]  # T
        self.punishment_multiplier = config["punishment_multiplier"]  # alpha
        self.welfare = config["welfare"]
        self.working_state = config["working_state"]
        self.verbose = config["verbose"]

        assert self.welfare in postprocessing.WELFARES, f"self.welfare: {self.welfare} must be in " \
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
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        self._select_witch_algo_to_use()
        actions, state_out, extra_fetches = self.algorithms[self.active_algo_idx].compute_actions(
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
        if self.working_state == WORKING_STATES[0] or self.working_state == WORKING_STATES[4]:
            self.active_algo_idx = OWN_COOP_POLICY_IDX
        elif self.working_state == WORKING_STATES[1] or self.working_state == WORKING_STATES[3]:
            self.active_algo_idx = OWN_SELFISH_POLICY_IDX
        elif self.working_state == WORKING_STATES[2]:
            self._select_algo_to_use_in_eval()
        else:
            raise ValueError(f'config["working_state"] must be one of {WORKING_STATES}')

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
        assert working_state_idx == OWN_COOP_POLICY_IDX or working_state_idx == OWN_SELFISH_POLICY_IDX, \
                    f"current working_state is {self.working_state} but you be one of " \
                    f"{[WORKING_STATES[OWN_COOP_POLICY_IDX], WORKING_STATES[OWN_SELFISH_POLICY_IDX]]}"

        self._update_lr_in_all_optimizers()

        # samples_copy = samples.copy()
        algo_to_train = self.algorithms[working_state_idx]
        learner_stats = {"learner_stats": {}}
        learner_stats["learner_stats"][f"learner_stats_algo{working_state_idx}"] = algo_to_train.learn_on_batch(
            samples)

        # if self.verbose > 0:
        #     for j, opt in enumerate(algo_to_train._optimizers):
        #         self._to_log[f"algo_{working_state_idx}_{j}_lr"] = [p["lr"]
        #                                                             for p in opt.param_groups][0]
        # self._to_log[f'algo{working_state_idx}_cur_lr'] = algo_to_train.cur_lr
        if self.verbose > 1:
            print(f"learn_on_batch WORKING_STATES {WORKING_STATES[working_state_idx]}, ")

        self._log_learning_rates()

        return learner_stats

    def _update_lr_in_all_optimizers(self):
        self.optimizer()

    def on_episode_step(self, opp_obs, last_obs, opp_action, worker,
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
            print('debit_threshold', self.debit_threshold)

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
            if self.verbose > 0:
                print("coop_opp_simulated_action != opp_action:",
                      coop_opp_simulated_action, opp_action)

            if worker.env.step_count_in_current_episode >= \
                    worker.env.max_steps:
                debit = 0
            else:
                debit = self._compute_debit(
                    last_obs, opp_action,  worker, base_env,
                    episode, env_index, coop_opp_simulated_action)
        else:
            if self.verbose > 0:
                print("coop_opp_simulated_action == opp_action")
            debit = 0
        self.total_debit += debit
        self._to_log['summed_debit'] = (
            debit + self._to_log['summed_debit']
            if 'summed_debit' in self._to_log else debit
        )
        if self.verbose > 0:
            print(f"debit {debit}")
            print(f"self.total_debit {self.total_debit}")


    def _is_starting_new_punishment_required(self):
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

    def on_episode_end(self):
        if self.working_state == WORKING_STATES[2]:
            self.total_debit = 0
            self.n_steps_to_punish = 0

    def _compute_debit(self, last_obs, opp_action, worker, base_env,
                       episode, env_index, coop_opp_simulated_action):
        raise NotImplementedError()

    def _compute_punishment_duration(
            self, opp_action,  coop_opp_simulated_action, worker, last_obs):
        raise NotImplementedError()




# TODO do the same in postprocessing (closure)
def get_amTFTCallBacks(additionnal_callbacks=[], **kwargs):

    class amTFTCallBacksPart(DefaultCallbacks):

        def on_episode_start(self, *, worker: "RolloutWorker",
                             base_env: BaseEnv,
                             policies: Dict[PolicyID, Policy],
                             episode: MultiAgentEpisode, env_index: int,
                             **kwargs):
            self._amTF_episode_starting = True
            self._amTF_temp_data = {}

        def on_episode_step(self, *, worker, base_env,
                            episode, env_index, **kwargs):

            agent_ids = list(worker.policy_map.keys())
            assert len(agent_ids) == 2, "Implemented for 2 players"

            for agent_id, policy in worker.policy_map.items():
                self._call_on_episode_step_from_policy(
                    agent_id, policy, agent_ids, episode, worker,
                    base_env, env_index)
            if self._amTF_episode_starting:
                self._amTF_episode_starting = False

        def _call_on_episode_step_from_policy(
                self, agent_id, policy, agent_ids, episode, worker,
                base_env, env_index):

            if self._is_callback_implemented_in_policy(
                    policy, 'on_episode_step'):
                opp_obs, raw_obs, opp_a = \
                    self.__extract_info_about_opponent(
                        agent_id, agent_ids, episode)
                if not self._amTF_episode_starting:
                    # Ignore the first call because the actions provided are
                    # fake (they were not played)
                    policy.on_episode_step(opp_obs, raw_obs, opp_a, worker,
                                           base_env, episode, env_index)

        def _is_callback_implemented_in_policy(self, policy, callback_method):
            return hasattr(policy, callback_method) and \
                   callable(getattr(policy, callback_method))

        def __extract_info_about_opponent(self, agent_id, agent_ids, episode):
            opp_agent_id = [one_id
                            for one_id in agent_ids
                            if one_id != agent_id][0]
            opp_a = episode.last_action_for(opp_agent_id)
            opp_obs, raw_obs \
                = self._load_observations_producing_the_last_action(
                    agent_id, episode, opp_agent_id)

            return opp_obs, raw_obs, opp_a

        def _load_observations_producing_the_last_action(
                self, agent_id, episode, opp_agent_id):
            """
            The episode object provides the last action with the given last
            observation produced by this action. But we need the
            observation that cause the agent to play this action
            thus the observations at t-1.
            """
            if not self._amTF_episode_starting:
                opp_obs = self._amTF_temp_data[agent_id][opp_agent_id]
                raw_obs = self._amTF_temp_data[agent_id]["raw_obs"]
            else:
                self._amTF_temp_data[agent_id] = {}
                opp_obs, raw_obs = None, None

            opp_new_obs = episode.last_observation_for(opp_agent_id)
            self._amTF_temp_data[agent_id][opp_agent_id] = opp_new_obs
            new_raw_obs = copy.deepcopy(episode._agent_to_last_raw_obs)
            self._amTF_temp_data[agent_id]["raw_obs"] = new_raw_obs

            return opp_obs, raw_obs

        def on_episode_end(self, *, worker, base_env,
                           policies, episode, env_index, **kwargs):

            for polidy_ID, policy in policies.items():
                if hasattr(policy, 'on_episode_end') and \
                        callable(policy.on_episode_end):
                    policy.on_episode_end()

        # def on_train_result(self, *, trainer, result: dict, **kwargs):
        #     self._share_weights_during_training(trainer)
        #
        # def _share_weights_during_training(self, trainer):
        #     local_policy_map = trainer.workers.local_worker().policy_map
        #     policy_ids = list(local_policy_map.keys())
        #     assert len(policy_ids) == 2
        #     policies_weights = None
        #     for i, policy_id in enumerate(policy_ids):
        #         opp_policy_id = policy_ids[(i + 1) % 2]
        #         if isinstance(local_policy_map[policy_id], amTFTPolicyBase):
        #             if local_policy_map[policy_id].working_state not in WORKING_STATES_IN_EVALUATION:
        #                 policy = local_policy_map[policy_id]
        #                 # Only get and set weights during training
        #                 if policies_weights is None:
        #                     policies_weights = trainer.get_weights()
        #                 assert isinstance(local_policy_map[opp_policy_id],
        #                                   amTFTPolicyBase), "if amTFT is training then " \
        #                                                      "the opponent must be " \
        #                                                      "using amTFT too"
        #                 # share weights during training of amTFT
        #                 policies_weights[policy_id][policy._nested_key(OPP_COOP_POLICY_IDX)] = \
        #                     policies_weights[opp_policy_id][policy._nested_key(OWN_COOP_POLICY_IDX)]
        #                 policies_weights[policy_id][policy._nested_key(OPP_SELFISH_POLICY_IDX)] = \
        #                     policies_weights[opp_policy_id][policy._nested_key(OWN_SELFISH_POLICY_IDX)]
        #     # Only get and set weights during training
        #     if policies_weights is not None:
        #         trainer.set_weights(policies_weights)

    if not isinstance(additionnal_callbacks, Iterable):
        additionnal_callbacks = [additionnal_callbacks]

    amTFTCallBacks = miscellaneous.merge_callbacks(  # WelfareAndPostprocessCallbacks,
        amTFTCallBacksPart,
        *additionnal_callbacks)

    return amTFTCallBacks

