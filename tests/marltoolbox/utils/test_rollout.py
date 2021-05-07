import copy
import os
import tempfile

import numpy as np
import time
from ray.rllib.agents.pg import PGTrainer, PGTorchPolicy
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR

from marltoolbox.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
)
from marltoolbox.examples.rllib_api.pg_ipd import get_rllib_config
from marltoolbox.utils import log, miscellaneous
from marltoolbox.utils import rollout

CONSTANT_REWARD = 1.0
EPI_LENGTH = 33


def test_rollout_actions_played_equal_actions_specified():
    policy_agent_mapping = lambda policy_id: policy_id
    assert_actions_played_equal_actions_specified(
        policy_agent_mapping,
        rollout_length=20,
        num_episodes=1,
        actions_list=[0, 1] * 100,
    )
    assert_actions_played_equal_actions_specified(
        policy_agent_mapping,
        rollout_length=40,
        num_episodes=1,
        actions_list=[1, 1] * 100,
    )
    assert_actions_played_equal_actions_specified(
        policy_agent_mapping,
        rollout_length=77,
        num_episodes=2,
        actions_list=[0, 0] * 100,
    )
    assert_actions_played_equal_actions_specified(
        policy_agent_mapping,
        rollout_length=77,
        num_episodes=3,
        actions_list=[0, 1] * 100,
    )
    assert_actions_played_equal_actions_specified(
        policy_agent_mapping,
        rollout_length=6,
        num_episodes=3,
        actions_list=[1, 0] * 100,
    )


def assert_actions_played_equal_actions_specified(
    policy_agent_mapping, rollout_length, num_episodes, actions_list
):
    rollout_results, worker = _when_perform_rollouts_wt_given_actions(
        actions_list, rollout_length, policy_agent_mapping, num_episodes
    )

    _assert_length_of_rollout(rollout_results, num_episodes, rollout_length)

    n_steps_in_last_epi, steps_in_last_epi = _compute_n_steps_in_last_epi(
        rollout_results, rollout_length, num_episodes
    )

    all_steps = _unroll_all_steps(rollout_results)

    # Verify that the actions played are the actions we forced to play
    _for_each_player_exec_fn(
        worker,
        _assert_played_the_actions_specified,
        all_steps,
        rollout_length,
        num_episodes,
        actions_list,
    )
    _for_each_player_exec_fn(
        worker,
        _assert_played_the_actions_specified_during_last_epi_only,
        all_steps,
        n_steps_in_last_epi,
        steps_in_last_epi,
        actions_list,
    )


def _when_perform_rollouts_wt_given_actions(
    actions_list, rollout_length, policy_agent_mapping, num_episodes
):
    worker = _init_worker(actions_list=actions_list)
    rollout_results = rollout.internal_rollout(
        worker,
        num_steps=rollout_length,
        policy_agent_mapping=policy_agent_mapping,
        reset_env_before=True,
        num_episodes=num_episodes,
    )
    return rollout_results, worker


class _FakeEnvWtCstReward(IteratedPrisonersDilemma):
    def step(self, actions: dict):
        observations, rewards, epi_is_done, info = super().step(actions)

        for k in rewards.keys():
            rewards[k] = CONSTANT_REWARD

        return observations, rewards, epi_is_done, info


def _make_fake_policy_wt_defined_actions(list_actions_to_play):
    class FakePolicyWtDefinedActions(PGTorchPolicy):
        def _compute_action_helper(self, *args, **kwargs):
            action = list_actions_to_play.pop(0)
            return np.array([action]), [], {}

        def _initialize_loss_from_dummy_batch(
            self,
            auto_remove_unneeded_view_reqs: bool = True,
            stats_fn=None,
        ) -> None:
            pass

    return FakePolicyWtDefinedActions


def _init_worker(actions_list=None):
    train_n_replicates = 1
    debug = True
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("testing")

    rllib_config, stop_config = get_rllib_config(seeds, debug)
    rllib_config["env"] = _FakeEnvWtCstReward
    rllib_config["env_config"]["max_steps"] = EPI_LENGTH
    rllib_config["seed"] = int(time.time())
    if actions_list is not None:
        for policy_id in _FakeEnvWtCstReward({}).players_ids:
            policy_to_modify = list(
                rllib_config["multiagent"]["policies"][policy_id]
            )
            policy_to_modify[0] = _make_fake_policy_wt_defined_actions(
                copy.deepcopy(actions_list)
            )
            rllib_config["multiagent"]["policies"][
                policy_id
            ] = policy_to_modify

    pg_trainer = PGTrainer(
        rllib_config, logger_creator=_get_logger_creator(exp_name)
    )
    return pg_trainer.workers._local_worker


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


def _for_each_player_exec_fn(worker, fn, *arg, **kwargs):
    for policy_id in worker.env.players_ids:
        fn(policy_id, *arg, **kwargs)


def _assert_played_the_actions_specified(
    policy_id, all_steps, rollout_length, num_episodes, actions_list
):
    actions_played = [step[1][policy_id] for step in all_steps]
    assert len(actions_played) == min(
        rollout_length, num_episodes * EPI_LENGTH
    )
    for action_required, action_played in zip(
        actions_list[: len(all_steps)], actions_played
    ):
        assert action_required == action_played


def _assert_played_the_actions_specified_during_last_epi_only(
    policy_id, all_steps, n_steps_in_last_epi, steps_in_last_epi, actions_list
):
    actions_played = [step[1][policy_id] for step in steps_in_last_epi]
    assert len(actions_played) == n_steps_in_last_epi
    actions_required_during_last_epi = actions_list[: len(all_steps)][
        -n_steps_in_last_epi:
    ]
    for action_required, action_played in zip(
        actions_required_during_last_epi, actions_played
    ):
        assert action_required == action_played


def _assert_length_of_rollout(rollout_results, num_episodes, rollout_length):
    assert (
        rollout_results._num_episodes == num_episodes
        or rollout_results._total_steps == rollout_length
    )


def _compute_n_steps_in_last_epi(
    rollout_results, rollout_length, num_episodes
):
    steps_in_last_epi = rollout_results._current_rollout
    if rollout_results._total_steps == rollout_length:
        n_steps_in_last_epi = rollout_results._total_steps % EPI_LENGTH
    elif rollout_results._num_episodes == num_episodes:
        n_steps_in_last_epi = EPI_LENGTH

    assert n_steps_in_last_epi == len(
        steps_in_last_epi
    ), f"{n_steps_in_last_epi} == {len(steps_in_last_epi)}"

    return n_steps_in_last_epi, steps_in_last_epi


def _unroll_all_steps(rollout_results):
    all_steps = []
    for epi_rollout in rollout_results._rollouts:
        all_steps.extend(epi_rollout)
    return all_steps


def test_rollout_rewards_received_equal_constant_reward():
    policy_agent_mapping = lambda policy_id: policy_id
    assert_rewards_received_are_rewards_specified(
        policy_agent_mapping, rollout_length=20, num_episodes=1
    )
    assert_rewards_received_are_rewards_specified(
        policy_agent_mapping, rollout_length=40, num_episodes=1
    )
    assert_rewards_received_are_rewards_specified(
        policy_agent_mapping, rollout_length=77, num_episodes=2
    )
    assert_rewards_received_are_rewards_specified(
        policy_agent_mapping, rollout_length=77, num_episodes=3
    )
    assert_rewards_received_are_rewards_specified(
        policy_agent_mapping, rollout_length=6, num_episodes=3
    )


def assert_rewards_received_are_rewards_specified(
    policy_agent_mapping, rollout_length, num_episodes
):
    rollout_results, worker = _when_perform_rollouts_wt_given_actions(
        None, rollout_length, policy_agent_mapping, num_episodes
    )

    _assert_length_of_rollout(rollout_results, num_episodes, rollout_length)

    n_steps_in_last_epi, steps_in_last_epi = _compute_n_steps_in_last_epi(
        rollout_results, rollout_length, num_episodes
    )

    all_steps = _unroll_all_steps(rollout_results)

    # Verify that the rewards received are the one we defined
    _for_each_player_exec_fn(
        worker,
        _assert_rewards_in_last_epi_are_as_specified,
        steps_in_last_epi,
        n_steps_in_last_epi,
    )

    _for_each_player_exec_fn(
        worker,
        _assert_rewards_are_as_defined,
        all_steps,
        rollout_length,
        num_episodes,
    )


def _assert_rewards_in_last_epi_are_as_specified(
    policy_id, steps_in_last_epi, n_steps_in_last_epi
):
    rewards = [step[3][policy_id] for step in steps_in_last_epi]
    assert sum(rewards) == n_steps_in_last_epi * CONSTANT_REWARD
    assert len(rewards) == n_steps_in_last_epi


def _assert_rewards_are_as_defined(
    policy_id, all_steps, rollout_length, num_episodes
):
    rewards = [step[3][policy_id] for step in all_steps]
    assert (
        sum(rewards)
        == min(rollout_length, num_episodes * EPI_LENGTH) * CONSTANT_REWARD
    )
    assert len(rewards) == min(rollout_length, num_episodes * EPI_LENGTH)
