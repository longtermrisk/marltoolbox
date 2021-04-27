import copy
import os
import tempfile
import time

import numpy as np
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR

from marltoolbox.algos import amTFT
from marltoolbox.algos.amTFT import base_policy, base
from marltoolbox.algos.amTFT.base import (
    DEFAULT_NESTED_POLICY_COOP,
    DEFAULT_NESTED_POLICY_SELFISH,
    WORKING_STATES,
)
from marltoolbox.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
)
from marltoolbox.experiments.rllib_api import amtft_various_env
from marltoolbox.utils import postprocessing, log
from test_base_policy import init_amTFT, generate_fake_discrete_actions


def test_compute_actions_overwrite():
    am_tft_policy, env = init_amTFT(
        policy_class=amTFT.AmTFTRolloutsTorchPolicy
    )

    fake_actions = generate_fake_discrete_actions(env)
    env.reset()
    observations, rewards, done, info = env.step(fake_actions)

    am_tft_policy.use_opponent_policies = True
    fake_actions, fake_state_out, fake_extra_fetches = "fake", "fake", "fake"
    fake_actions_2nd, fake_state_out_2nd, fake_extra_fetches_2nd = (
        "fake_2nd",
        "fake_2nd",
        "fake_2nd",
    )
    am_tft_policy.overwrite_action = [
        (fake_actions, fake_state_out, fake_extra_fetches),
        (fake_actions_2nd, fake_state_out_2nd, fake_extra_fetches_2nd),
    ]
    actions, state_out, extra_fetches = am_tft_policy._compute_action_helper(
        observations[env.players_ids[0]],
        state_batches=None,
        seq_lens=1,
        explore=True,
        timestep=0,
    )
    assert actions == fake_actions
    assert state_out == fake_state_out
    assert extra_fetches == fake_extra_fetches
    actions, state_out, extra_fetches = am_tft_policy._compute_action_helper(
        observations[env.players_ids[0]],
        state_batches=None,
        seq_lens=1,
        explore=True,
        timestep=0,
    )
    assert actions == fake_actions_2nd
    assert state_out == fake_state_out_2nd
    assert extra_fetches == fake_extra_fetches_2nd


def test__select_algo_to_use_in_eval():
    am_tft_policy, env = init_amTFT(
        policy_class=amTFT.AmTFTRolloutsTorchPolicy
    )

    def assert_active_algo_idx(working_state_idx, active_algo_idx):
        am_tft_policy.working_state = base_policy.WORKING_STATES[
            working_state_idx
        ]
        am_tft_policy._select_witch_algo_to_use(None)
        assert (
            am_tft_policy.active_algo_idx == active_algo_idx
        ), f"{am_tft_policy.active_algo_idx} == {active_algo_idx}"

    am_tft_policy.use_opponent_policies = False
    am_tft_policy.n_steps_to_punish = 0
    assert_active_algo_idx(
        working_state_idx=2, active_algo_idx=base.OWN_COOP_POLICY_IDX
    )
    am_tft_policy.use_opponent_policies = False
    am_tft_policy.n_steps_to_punish = 1
    assert_active_algo_idx(
        working_state_idx=2, active_algo_idx=base.OWN_SELFISH_POLICY_IDX
    )

    am_tft_policy.use_opponent_policies = True
    am_tft_policy.performing_rollouts = True
    am_tft_policy.n_steps_to_punish_opponent = 0
    assert_active_algo_idx(
        working_state_idx=2, active_algo_idx=base.OPP_COOP_POLICY_IDX
    )
    am_tft_policy.use_opponent_policies = True
    am_tft_policy.performing_rollouts = True
    am_tft_policy.n_steps_to_punish_opponent = 1
    assert_active_algo_idx(
        working_state_idx=2, active_algo_idx=base.OPP_SELFISH_POLICY_IDX
    )


def test__duration_found_or_continue_search():
    am_tft_policy, env = init_amTFT(
        policy_class=amTFT.AmTFTRolloutsTorchPolicy
    )

    def assert_(k_to_explore, k_assert):
        (
            new_k_to_explore,
            continue_to_search_k,
        ) = am_tft_policy._duration_found_or_continue_search(k_to_explore)
        assert new_k_to_explore == k_assert
        assert continue_to_search_k == (k_to_explore != k_assert)

    am_tft_policy.punishment_debit = 2.0
    am_tft_policy.k_opp_loss = {0: 0.0, 1: 1.0}
    am_tft_policy.last_n_steps_played = 1.0
    assert_(k_to_explore=1, k_assert=1)
    am_tft_policy.last_n_steps_played = 3.0
    assert_(k_to_explore=1, k_assert=2)
    am_tft_policy.punishment_debit = 1.0
    assert_(k_to_explore=1, k_assert=1)

    am_tft_policy.punishment_debit = 6.0
    am_tft_policy.k_opp_loss = {4: 0.0, 5: 5.0}
    am_tft_policy.last_n_steps_played = 4.0
    assert_(k_to_explore=5, k_assert=5)
    am_tft_policy.last_n_steps_played = 6.0
    assert_(k_to_explore=5, k_assert=6)
    am_tft_policy.punishment_debit = 4.0
    assert_(k_to_explore=5, k_assert=5)
    am_tft_policy.k_opp_loss = {4: 4.0, 5: 5.0}
    am_tft_policy.punishment_debit = 3.0
    assert_(k_to_explore=5, k_assert=4)

    am_tft_policy.punishment_debit = 6.0
    am_tft_policy.k_opp_loss = {38: 37.5, 39: 39.0, 40: 40.0}
    am_tft_policy.last_n_steps_played = 45
    assert_(k_to_explore=40, k_assert=38)
    assert_(k_to_explore=39, k_assert=38)
    am_tft_policy.last_n_steps_played = 40
    assert_(k_to_explore=40, k_assert=38)
    am_tft_policy.punishment_debit = 41.0
    am_tft_policy.last_n_steps_played = 40
    assert_(k_to_explore=40, k_assert=40)
    assert_(k_to_explore=39, k_assert=40)
    am_tft_policy.last_n_steps_played = 10
    am_tft_policy.punishment_debit = 38.0
    assert_(k_to_explore=40, k_assert=10)
    am_tft_policy.last_n_steps_played = 40
    assert_(k_to_explore=40, k_assert=38)
    am_tft_policy.punishment_debit = 41.0
    assert_(k_to_explore=40, k_assert=40)
    assert_(k_to_explore=39, k_assert=40)
    am_tft_policy.punishment_debit = 39.5
    assert_(k_to_explore=40, k_assert=40)


class FakeEnvWtActionAsReward(IteratedPrisonersDilemma):
    def step(self, actions: dict):
        observations, rewards, epi_is_done, info = super().step(actions)

        for k in rewards.keys():
            rewards[k] = actions[k]

        return observations, rewards, epi_is_done, info


def make_fake_policy_class_wt_defined_actions(
    list_actions_to_play, ParentPolicyCLass
):
    class FakePolicyWtDefinedActions(ParentPolicyCLass):
        def _compute_action_helper(self, *args, **kwargs):
            print("len", len(list_actions_to_play))
            action = list_actions_to_play.pop(0)
            return np.array([action]), [], {}

        def _initialize_loss_from_dummy_batch(
            self,
            auto_remove_unneeded_view_reqs: bool = True,
            stats_fn=None,
        ) -> None:
            pass

    return FakePolicyWtDefinedActions


#
def init_worker(
    n_rollout_replicas,
    max_steps,
    actions_list_0=None,
    actions_list_1=None,
    actions_list_2=None,
    actions_list_3=None,
):
    train_n_replicates = 1
    debug = True
    exp_name, _ = log.log_in_current_day_dir("testing")

    hparams = amtft_various_env.get_hyperparameters(
        debug,
        train_n_replicates,
        filter_utilitarian=False,
        env="IteratedPrisonersDilemma",
    )

    stop, env_config, rllib_config = amtft_various_env.get_rllib_config(
        hparams, welfare_fn=postprocessing.WELFARE_UTILITARIAN
    )

    rllib_config["env"] = FakeEnvWtActionAsReward
    rllib_config["env_config"]["max_steps"] = max_steps
    rllib_config = _remove_dynamic_values_from_config(
        rllib_config, hparams, env_config, stop
    )

    for policy_id in FakeEnvWtActionAsReward({}).players_ids:
        policy_to_modify = list(
            rllib_config["multiagent"]["policies"][policy_id]
        )
        policy_to_modify[3]["rollout_length"] = max_steps
        policy_to_modify[3]["n_rollout_replicas"] = n_rollout_replicas
        policy_to_modify[3]["verbose"] = 1
        if actions_list_0 is not None:
            policy_to_modify[3]["nested_policies"][0][
                "Policy_class"
            ] = make_fake_policy_class_wt_defined_actions(
                copy.deepcopy(actions_list_0), DEFAULT_NESTED_POLICY_COOP
            )
        if actions_list_1 is not None:
            policy_to_modify[3]["nested_policies"][1][
                "Policy_class"
            ] = make_fake_policy_class_wt_defined_actions(
                copy.deepcopy(actions_list_1), DEFAULT_NESTED_POLICY_SELFISH
            )
        if actions_list_2 is not None:
            policy_to_modify[3]["nested_policies"][2][
                "Policy_class"
            ] = make_fake_policy_class_wt_defined_actions(
                copy.deepcopy(actions_list_2), DEFAULT_NESTED_POLICY_COOP
            )
        if actions_list_3 is not None:
            policy_to_modify[3]["nested_policies"][3][
                "Policy_class"
            ] = make_fake_policy_class_wt_defined_actions(
                copy.deepcopy(actions_list_3), DEFAULT_NESTED_POLICY_SELFISH
            )
        rllib_config["multiagent"]["policies"][policy_id] = tuple(
            policy_to_modify
        )
    rllib_config["exploration_config"]["temperature_schedule"] = rllib_config[
        "exploration_config"
    ]["temperature_schedule"].func(rllib_config)
    import ray

    ray.tune.sample
    dqn_trainer = DQNTrainer(
        rllib_config, logger_creator=_get_logger_creator(exp_name)
    )
    worker = dqn_trainer.workers._local_worker

    am_tft_policy_row = worker.get_policy("player_row")
    am_tft_policy_col = worker.get_policy("player_col")
    am_tft_policy_row.working_state = WORKING_STATES[2]
    am_tft_policy_col.working_state = WORKING_STATES[2]
    print("env setup")

    return worker, am_tft_policy_row, am_tft_policy_col


def _remove_dynamic_values_from_config(
    rllib_config, hparams, env_config, stop
):
    rllib_config["seed"] = int(time.time())
    rllib_config["learning_starts"] = int(
        rllib_config["env_config"]["max_steps"]
        * rllib_config["env_config"]["bs_epi_mul"]
    )
    rllib_config["buffer_size"] = int(
        env_config["max_steps"]
        * env_config["buf_frac"]
        * stop["episodes_total"]
    )
    rllib_config["train_batch_size"] = int(
        env_config["max_steps"] * env_config["bs_epi_mul"]
    )
    rllib_config["training_intensity"] = int(
        rllib_config["num_envs_per_worker"]
        * rllib_config["num_workers"]
        * hparams["training_intensity"]
    )
    return rllib_config


def _get_logger_creator(exp_name):
    logdir_prefix = exp_name + "/"
    tail, head = os.path.split(exp_name)
    tail_bis, _ = os.path.split(tail)

    def default_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(DEFAULT_RESULTS_DIR):
            os.makedirs(DEFAULT_RESULTS_DIR)
        if not os.path.exists(os.path.join(DEFAULT_RESULTS_DIR, tail_bis)):
            os.mkdir(os.path.join(DEFAULT_RESULTS_DIR, tail_bis))
        if not os.path.exists(os.path.join(DEFAULT_RESULTS_DIR, tail)):
            os.mkdir(os.path.join(DEFAULT_RESULTS_DIR, tail))
        if not os.path.exists(os.path.join(DEFAULT_RESULTS_DIR, exp_name)):
            os.mkdir(os.path.join(DEFAULT_RESULTS_DIR, exp_name))
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR
        )
        return UnifiedLogger(config, logdir, loggers=None)

    return default_logger_creator


def test__compute_debit_using_rollouts():
    def assert_debit_value_computed(
        worker_, am_tft_policy, last_obs, opp_action, assert_debit
    ):
        worker_.foreach_env(lambda env: env.reset())
        debit = am_tft_policy._compute_debit_using_rollouts(
            last_obs, opp_action, worker_
        )
        assert debit == assert_debit

    # Never giving reward except for the opp first action
    def init_no_extra_reward(max_steps_):
        n_rollout_replicas = 2
        worker_, am_tft_policy_row_, am_tft_policy_col_ = init_worker(
            n_rollout_replicas=n_rollout_replicas,
            max_steps=max_steps_,
            # n steps x 2 rollouts x n_rollout_replicas//2
            actions_list_0=[0] * (max_steps_ * 2 * n_rollout_replicas // 2),
            actions_list_1=[0] * (max_steps_ * 2 * n_rollout_replicas // 2),
            actions_list_2=[0] * (max_steps_ * 2 * n_rollout_replicas // 2),
            actions_list_3=[0] * (max_steps_ * 2 * n_rollout_replicas // 2),
        )
        return worker_, am_tft_policy_row_, am_tft_policy_col_

    max_steps = 2
    worker, am_tft_policy_row, am_tft_policy_col = init_no_extra_reward(
        max_steps
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_row,
        {"player_row": 0, "player_col": 0},
        opp_action=0,
        assert_debit=0,
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_col,
        {"player_row": 1, "player_col": 0},
        opp_action=1,
        assert_debit=1,
    )

    worker, am_tft_policy_row, am_tft_policy_col = init_no_extra_reward(
        max_steps
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_row,
        {"player_row": 1, "player_col": 0},
        opp_action=1,
        assert_debit=1,
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_col,
        {"player_row": 1, "player_col": 1},
        opp_action=0,
        assert_debit=0,
    )

    # actions_list_3 (opp selfish) should never be used here
    def init_selfish_opp_advantaged(max_steps):
        n_rollout_replicas = 2
        worker, am_tft_policy_row, am_tft_policy_col = init_worker(
            n_rollout_replicas=n_rollout_replicas,
            max_steps=max_steps,
            # n steps x 2 rollouts x n_rollout_replicas//2
            actions_list_0=[0] * (max_steps * 2 * n_rollout_replicas // 2),
            actions_list_1=[0] * (max_steps * 2 * n_rollout_replicas // 2),
            actions_list_2=[0] * (max_steps * 2 * n_rollout_replicas // 2),
            actions_list_3=[1] * (max_steps * 2 * n_rollout_replicas // 2),
        )
        return worker, am_tft_policy_row, am_tft_policy_col

    max_steps = 2
    worker, am_tft_policy_row, am_tft_policy_col = init_selfish_opp_advantaged(
        max_steps
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_row,
        {"player_row": 0, "player_col": 0},
        opp_action=0,
        assert_debit=0,
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_col,
        {"player_row": 1, "player_col": 0},
        opp_action=1,
        assert_debit=1,
    )

    # coop opp would have all get a reward of 1
    def init_coop_opp_advantaged(max_steps):
        n_rollout_replicas = 2
        worker, am_tft_policy_row, am_tft_policy_col = init_worker(
            n_rollout_replicas=n_rollout_replicas,
            max_steps=max_steps,
            # n steps x 2 rollouts x n_rollout_replicas//2
            actions_list_0=[0] * (max_steps * 2 * n_rollout_replicas // 2),
            actions_list_1=[0] * (max_steps * 2 * n_rollout_replicas // 2),
            actions_list_2=[1] * (max_steps * 2 * n_rollout_replicas // 2),
            actions_list_3=[0] * (max_steps * 2 * n_rollout_replicas // 2),
        )
        return worker, am_tft_policy_row, am_tft_policy_col

    max_steps = 3
    worker, am_tft_policy_row, am_tft_policy_col = init_coop_opp_advantaged(
        max_steps
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_row,
        {"player_row": 1, "player_col": 0},
        opp_action=1,
        assert_debit=0,
    )
    assert_debit_value_computed(
        worker,
        am_tft_policy_col,
        {"player_row": 1, "player_col": 1},
        opp_action=0,
        assert_debit=-1,
    )


def test__compute_punishment_duration_from_rollouts():
    def assert_(
        worker_,
        am_tft_policy,
        last_obs,
        assert_k,
        total_debit,
        punishment_multiplier,
        last_k,
        step_count_in_current_episode=0,
    ):
        worker_.foreach_env(lambda env: env.reset())
        worker.env.step_count_in_current_episode = (
            step_count_in_current_episode
        )
        am_tft_policy.last_k = last_k
        am_tft_policy.total_debit = total_debit
        am_tft_policy.punishment_multiplier = punishment_multiplier
        k = am_tft_policy._compute_punishment_duration_from_rollouts(
            worker_, last_obs
        )
        print("k, assert_k, last_k", k, assert_k, last_k)
        assert k == assert_k

    # Never giving reward
    def init_wt_no_extra_reward(max_steps_):
        n_rollout_replicas = 2
        worker, am_tft_policy_row, am_tft_policy_col = init_worker(
            n_rollout_replicas=n_rollout_replicas,
            max_steps=max_steps_,
            # n steps x 2 rollouts x n_rollout_replicas//2
            actions_list_0=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_1=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_2=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_3=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
        )
        return worker, am_tft_policy_row, am_tft_policy_col

    max_steps = 10
    for last_k in range(1, max_steps, 1):
        worker, am_tft_policy_row, am_tft_policy_col = init_wt_no_extra_reward(
            max_steps_=max_steps
        )
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 1, "player_col": 1},
            assert_k=max(last_k - 1, 0),
            total_debit=1,
            punishment_multiplier=2,
            last_k=last_k,
        )

    def init_wt_reward_for_coop_opp(max_steps_):
        n_rollout_replicas = 2
        worker, am_tft_policy_row, am_tft_policy_col = init_worker(
            n_rollout_replicas=n_rollout_replicas,
            max_steps=max_steps_,
            # n steps x 2 rollouts x n_rollout_replicas//2
            actions_list_0=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_1=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_2=[1]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_3=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
        )
        return worker, am_tft_policy_row, am_tft_policy_col

    max_steps = 10
    for last_k in range(1, max_steps, 1):
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_coop_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 1, "player_col": 1},
            assert_k=2,
            total_debit=1,
            punishment_multiplier=2,
            last_k=last_k,
        )
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_coop_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 0, "player_col": 1},
            assert_k=1,
            total_debit=0.25,
            punishment_multiplier=2,
            last_k=last_k,
        )
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_coop_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 0, "player_col": 1},
            assert_k=5,
            total_debit=1,
            punishment_multiplier=5,
            last_k=last_k,
        )
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_coop_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 0, "player_col": 1},
            assert_k=6,
            total_debit=1,
            punishment_multiplier=5.1,
            last_k=last_k,
        )
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_coop_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 0, "player_col": 1},
            assert_k=5,
            total_debit=1,
            punishment_multiplier=4.9,
            last_k=last_k,
        )

    def init_wt_reward_for_self_opp(max_steps_):
        n_rollout_replicas = 2
        worker, am_tft_policy_row, am_tft_policy_col = init_worker(
            n_rollout_replicas=n_rollout_replicas,
            max_steps=max_steps_,
            # n steps x 2 rollouts x n_rollout_replicas//2
            actions_list_0=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_1=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_2=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_3=[1]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
        )
        return worker, am_tft_policy_row, am_tft_policy_col

    max_steps = 10
    for last_k in range(1, max_steps, 1):
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_self_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 1, "player_col": 1},
            assert_k=max(last_k - 1, 0),
            total_debit=1,
            punishment_multiplier=2,
            last_k=last_k,
        )
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_self_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 1, "player_col": 1},
            assert_k=max(last_k - 1, 0),
            total_debit=0.1,
            punishment_multiplier=2,
            last_k=last_k,
        )

    def init_wt_reward_for_self_and_coop_opp(max_steps_):
        n_rollout_replicas = 2
        worker, am_tft_policy_row, am_tft_policy_col = init_worker(
            n_rollout_replicas=n_rollout_replicas,
            max_steps=max_steps_,
            # n steps x 2 rollouts x n_rollout_replicas//2
            actions_list_0=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_1=[0]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_2=[1]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
            actions_list_3=[1]
            * (max_steps_ * 2 * n_rollout_replicas // 2 * 100),
        )
        return worker, am_tft_policy_row, am_tft_policy_col

    max_steps = 10
    for last_k in range(max_steps):
        (
            worker,
            am_tft_policy_row,
            am_tft_policy_col,
        ) = init_wt_reward_for_self_and_coop_opp(max_steps_=max_steps)
        assert_(
            worker,
            am_tft_policy_row,
            last_obs={"player_row": 1, "player_col": 1},
            assert_k=max(last_k - 1, 0),
            total_debit=1,
            punishment_multiplier=2,
            last_k=last_k,
        )

    max_steps = 5
    for i in range(max_steps):
        for last_k in range(max_steps):
            (
                worker,
                am_tft_policy_row,
                am_tft_policy_col,
            ) = init_wt_reward_for_coop_opp(max_steps_=max_steps)
            assert_(
                worker,
                am_tft_policy_row,
                last_obs={"player_row": 1, "player_col": 1},
                assert_k=int(min(1.5 * 2, max_steps - i)),
                total_debit=1.5,
                punishment_multiplier=2,
                last_k=last_k,
                step_count_in_current_episode=i,
            )
