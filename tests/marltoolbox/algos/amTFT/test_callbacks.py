# import copy
#
# import numpy as np
# import time
# from ray.rllib.agents.dqn import DQNTrainer
#
# from marltoolbox.algos import amTFT
# from marltoolbox.algos.amTFT import base_policy
# from marltoolbox.algos.amTFT.base_policy import DEFAULT_NESTED_POLICY_COOP, \
#     DEFAULT_NESTED_POLICY_SELFISH, WORKING_STATES
# from marltoolbox.envs.matrix_sequential_social_dilemma import \
#     IteratedPrisonersDilemma
# from marltoolbox.examples.rllib_api.amtft_various_env import get_rllib_config, \
#     modify_hyperparams_for_the_selected_env, get_hyperparameters
# from marltoolbox.utils import log, miscellaneous
# from marltoolbox.utils import postprocessing
#
# from test_amTFTRolloutsTorchPolicy import FakeEnvWtActionAsReward, \
#     make_FakePolicyWtDefinedActions
#
# amTFTCallback =amTFT.get_amTFTCallBacks(
#     additionnal_callbacks=[
#         log.get_logging_callbacks_class(),
#         # This only overwrite the reward that is used for training
#         # not the one in the metrics
#         postprocessing.OverwriteRewardWtWelfareCallback]
# )
#
# class CallbackAssert(amTFTCallback):
#
#     def on_episode_step(self, *, worker, base_env,
#                         episode, env_index, **kwargs):
#         super()
#     def on_episode_end(self, *, worker, base_env,
#                        policies, episode, env_index, **kwargs):
#
#         for polidy_ID, policy in policies.items():
#             if self._is_callback_implemented_in_policy(
#                     policy, 'on_episode_end'):
#                 policy.on_episode_end()
#
#     def on_train_result(self, *, trainer, result: dict, **kwargs):
#         self._share_weights_during_training(trainer)
#
#
# def init_worker(n_rollout_replicas,
#                 max_steps,
#                 actions_list_0=None,
#                 actions_list_1=None,
#                 actions_list_2=None,
#                 actions_list_3=None,
#                 ):
#     train_n_replicates = 1
#     debug = True
#     exp_name, _ = log.log_in_current_day_dir("testing")
#
#     hparams = get_hyperparameters(
#         debug, train_n_replicates, filter_utilitarian=False,
#         env="IteratedPrisonersDilemma")
#
#     _, _, rllib_config = \
#         get_rllib_config(
#             hparams,
#             welfare_fn=postprocessing.WELFARE_UTILITARIAN)
#
#     rllib_config['callbacks'] =
#     rllib_config['env'] = FakeEnvWtActionAsReward
#     rllib_config['env_config']['max_steps'] = max_steps
#     rllib_config['seed'] = int(time.time())
#     for policy_id in FakeEnvWtActionAsReward({}).players_ids:
#         policy_to_modify = list(rllib_config['multiagent']["policies"][policy_id])
#         policy_to_modify[3]["rollout_length"] = max_steps
#         policy_to_modify[3]["n_rollout_replicas"] = n_rollout_replicas
#         policy_to_modify[3]["verbose"] = 1
#         if actions_list_0 is not None:
#             policy_to_modify[3]["nested_policies"][0]["Policy_class"] = \
#                 make_FakePolicyWtDefinedActions(copy.deepcopy(actions_list_0),
#                                                 DEFAULT_NESTED_POLICY_COOP)
#         if actions_list_1 is not None:
#             policy_to_modify[3]["nested_policies"][1]["Policy_class"] = \
#                 make_FakePolicyWtDefinedActions(copy.deepcopy(actions_list_1),
#                                                 DEFAULT_NESTED_POLICY_SELFISH)
#         if actions_list_2 is not None:
#             policy_to_modify[3]["nested_policies"][2]["Policy_class"] = \
#                 make_FakePolicyWtDefinedActions(copy.deepcopy(actions_list_2),
#                                                 DEFAULT_NESTED_POLICY_COOP)
#         if actions_list_3 is not None:
#             policy_to_modify[3]["nested_policies"][3]["Policy_class"] = \
#                 make_FakePolicyWtDefinedActions(copy.deepcopy(actions_list_3),
#                                                 DEFAULT_NESTED_POLICY_SELFISH)
#         rllib_config['multiagent']["policies"][policy_id] = \
#             tuple(policy_to_modify)
#
#     dqn_trainer = DQNTrainer(rllib_config)
#     worker = dqn_trainer.workers._local_worker
#
#     am_tft_policy_row = worker.get_policy("player_row")
#     am_tft_policy_col = worker.get_policy("player_col")
#     am_tft_policy_row.working_state = WORKING_STATES[2]
#     am_tft_policy_col.working_state = WORKING_STATES[2]
#
#     return worker, am_tft_policy_row, am_tft_policy_col
#
# def test__compute_debit_using_rollouts():
#     def assert_(worker_, am_tft_policy, last_obs, opp_action, assert_debit):
#         worker_.foreach_env(lambda env: env.reset())
#         debit = am_tft_policy._compute_debit_using_rollouts(last_obs, opp_action, worker_)
#         assert debit == assert_debit
#
#     # Never giving reward except for the opp first action
#     def init_no_extra_reward(max_steps_):
#         n_rollout_replicas = 2
#         worker_, am_tft_policy_row_, am_tft_policy_col_ = init_worker(
#             n_rollout_replicas=n_rollout_replicas,
#             max_steps=max_steps_,
#             # n steps x 2 rollouts x n_rollout_replicas//2
#             actions_list_0=[0] * (max_steps_ * 2 * n_rollout_replicas // 2),
#             actions_list_1=[0] * (max_steps_ * 2 * n_rollout_replicas // 2),
#             actions_list_2=[0] * (max_steps_ * 2 * n_rollout_replicas // 2),
#             actions_list_3=[0] * (max_steps_ * 2 * n_rollout_replicas // 2))
#         return worker_, am_tft_policy_row_, am_tft_policy_col_
#
#     max_steps = 2
#     worker, am_tft_policy_row, am_tft_policy_col = \
#         init_no_extra_reward(max_steps)
#     assert_(worker, am_tft_policy_row,
#             {"player_row": 0, "player_col": 0},
#             opp_action=0,
#             assert_debit=0)
#     assert_(worker, am_tft_policy_col,
#             {"player_row": 1, "player_col": 0},
#             opp_action=1,
#             assert_debit=1)
#
#     worker, am_tft_policy_row, am_tft_policy_col = \
#         init_no_extra_reward(max_steps)
#     assert_(worker, am_tft_policy_row,
#             {"player_row": 1, "player_col": 0},
#             opp_action=1,
#             assert_debit=1)
#     assert_(worker, am_tft_policy_col,
#             {"player_row": 1, "player_col": 1},
#             opp_action=0,
#             assert_debit=0)
#     #
#     # # actions_list_3 (opp selfish) should never be used here
#     # def init_selfish_opp_advantaged(max_steps):
#     #     n_rollout_replicas = 2
#     #     worker, am_tft_policy_row, am_tft_policy_col = init_worker(
#     #         n_rollout_replicas=n_rollout_replicas,
#     #         max_steps=max_steps,
#     #         # n steps x 2 rollouts x n_rollout_replicas//2
#     #         actions_list_0=[0] * (max_steps * 2 * n_rollout_replicas // 2),
#     #         actions_list_1=[0] * (max_steps * 2 * n_rollout_replicas // 2),
#     #         actions_list_2=[0] * (max_steps * 2 * n_rollout_replicas // 2),
#     #         actions_list_3=[1] * (max_steps * 2 * n_rollout_replicas // 2))
#     #     return worker, am_tft_policy_row, am_tft_policy_col
#     #
#     # max_steps = 2
#     # worker, am_tft_policy_row, am_tft_policy_col = \
#     #     init_selfish_opp_advantaged(max_steps)
#     # assert_(worker, am_tft_policy_row,
#     #         {"player_row": 0, "player_col": 0},
#     #         opp_action=0,
#     #         assert_debit=0)
#     # assert_(worker, am_tft_policy_col,
#     #         {"player_row": 1, "player_col": 0},
#     #         opp_action=1,
#     #         assert_debit=1)
#     #
#     # # coop opp would have all get a reward of 1
#     # def init_coop_opp_advantaged(max_steps):
#     #     n_rollout_replicas = 2
#     #     worker, am_tft_policy_row, am_tft_policy_col = init_worker(
#     #         n_rollout_replicas=n_rollout_replicas,
#     #         max_steps=max_steps,
#     #         # n steps x 2 rollouts x n_rollout_replicas//2
#     #         actions_list_0=[0] * (max_steps * 2 * n_rollout_replicas // 2),
#     #         actions_list_1=[0] * (max_steps * 2 * n_rollout_replicas // 2),
#     #         actions_list_2=[1] * (max_steps * 2 * n_rollout_replicas // 2),
#     #         actions_list_3=[0] * (max_steps * 2 * n_rollout_replicas // 2))
#     #     return worker, am_tft_policy_row, am_tft_policy_col
#     #
#     # max_steps = 3
#     # worker, am_tft_policy_row, am_tft_policy_col = \
#     #     init_coop_opp_advantaged(max_steps)
#     # assert_(worker, am_tft_policy_row,
#     #         {"player_row": 1, "player_col": 0},
#     #         opp_action=1,
#     #         assert_debit=0)
#     # assert_(worker, am_tft_policy_col,
#     #         {"player_row": 1, "player_col": 1},
#     #         opp_action=0,
#     #         assert_debit=-1)