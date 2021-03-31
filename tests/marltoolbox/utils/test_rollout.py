import copy
import time
import os
import tempfile


import numpy as np
from ray.rllib.agents.pg import PGTrainer, PGTorchPolicy
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR

from marltoolbox.envs.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma
from marltoolbox.examples.rllib_api.pg_ipd import get_rllib_config
from marltoolbox.utils import log, miscellaneous
from marltoolbox.utils import rollout

CONSTANT_REWARD = 1.0
EPI_LENGTH = 33


class FakeEnvWtCstReward(IteratedPrisonersDilemma):

    def step(self, actions: dict):
        observations, rewards, epi_is_done, info = super().step(actions)

        for k in rewards.keys():
            rewards[k] = CONSTANT_REWARD

        return observations, rewards, epi_is_done, info


def make_FakePolicyWtDefinedActions(list_actions_to_play):
    class FakePolicyWtDefinedActions(PGTorchPolicy):
        def compute_actions(self, *args, **kwargs):
            action = list_actions_to_play.pop(0)
            return np.array([action]), [], {}

    return FakePolicyWtDefinedActions


def init_worker(actions_list=None):
    train_n_replicates = 1
    debug = True
    stop_iters = 200
    tf = False
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("testing")

    rllib_config, stop_config = get_rllib_config(seeds, debug, stop_iters, tf)
    rllib_config['env'] = FakeEnvWtCstReward
    rllib_config['env_config']['max_steps'] = EPI_LENGTH
    rllib_config['seed'] = int(time.time())
    if actions_list is not None:
        for policy_id in FakeEnvWtCstReward({}).players_ids:
            policy_to_modify = list(rllib_config['multiagent']["policies"][policy_id])
            policy_to_modify[0] = make_FakePolicyWtDefinedActions(copy.deepcopy(actions_list))
            rllib_config['multiagent']["policies"][policy_id] = policy_to_modify

    pg_trainer = PGTrainer(rllib_config, logger_creator=_get_logger_creator(exp_name))
    return pg_trainer.workers._local_worker

def _get_logger_creator(exp_name):
    logdir_prefix = exp_name + '/'
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
        if not os.path.exists(os.path.join(DEFAULT_RESULTS_DIR,exp_name)):
            os.mkdir(os.path.join(DEFAULT_RESULTS_DIR,exp_name))
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)
        return UnifiedLogger(config, logdir, loggers=None)
    return default_logger_creator

def test_rollout_constant_reward():
    policy_agent_mapping = (lambda policy_id: policy_id)

    def assert_(rollout_length, num_episodes):
        worker = init_worker()
        rollout_results = rollout.internal_rollout(worker,
                                                   num_steps=rollout_length,
                                                   policy_agent_mapping=policy_agent_mapping,
                                                   reset_env_before=True,
                                                   num_episodes=num_episodes)
        assert rollout_results._num_episodes == num_episodes or rollout_results._total_steps == rollout_length

        steps_in_last_epi = rollout_results._current_rollout
        if rollout_results._total_steps == rollout_length:
            n_steps_in_last_epi = rollout_results._total_steps % EPI_LENGTH
        elif rollout_results._num_episodes == num_episodes:
            n_steps_in_last_epi = EPI_LENGTH

        # Verify rewards
        for policy_id in worker.env.players_ids:
            rewards = [step[3][policy_id] for step in steps_in_last_epi]
            assert sum(rewards) == n_steps_in_last_epi * CONSTANT_REWARD
            assert len(rewards) == n_steps_in_last_epi
        all_steps = []
        for epi_rollout in rollout_results._rollouts:
            all_steps.extend(epi_rollout)
        for policy_id in worker.env.players_ids:
            rewards = [step[3][policy_id] for step in all_steps]
            assert sum(rewards) == min(rollout_length, num_episodes * EPI_LENGTH) * CONSTANT_REWARD
            assert len(rewards) == min(rollout_length, num_episodes * EPI_LENGTH)

    assert_(rollout_length=20, num_episodes=1)
    assert_(rollout_length=40, num_episodes=1)
    assert_(rollout_length=77, num_episodes=2)
    assert_(rollout_length=77, num_episodes=3)
    assert_(rollout_length=6, num_episodes=3)


def test_rollout_specified_actions():
    policy_agent_mapping = (lambda policy_id: policy_id)

    def assert_(rollout_length, num_episodes, actions_list):
        worker = init_worker(actions_list=actions_list)
        rollout_results = rollout.internal_rollout(worker,
                                                   num_steps=rollout_length,
                                                   policy_agent_mapping=policy_agent_mapping,
                                                   reset_env_before=True,
                                                   num_episodes=num_episodes)
        assert rollout_results._num_episodes == num_episodes or rollout_results._total_steps == rollout_length

        steps_in_last_epi = rollout_results._current_rollout
        if rollout_results._total_steps == rollout_length:
            n_steps_in_last_epi = rollout_results._total_steps % EPI_LENGTH
        elif rollout_results._num_episodes == num_episodes:
            n_steps_in_last_epi = EPI_LENGTH

        # Verify actions
        all_steps = []
        for epi_rollout in rollout_results._rollouts:
            all_steps.extend(epi_rollout)
        for policy_id in worker.env.players_ids:
            actions_played = [step[1][policy_id] for step in all_steps]
            assert len(actions_played) == min(rollout_length, num_episodes * EPI_LENGTH)
            print(actions_list[1:1 + len(all_steps)], actions_played)
            for action_required, action_played in zip(actions_list[:len(all_steps)], actions_played):
                assert action_required == action_played
        for policy_id in worker.env.players_ids:
            actions_played = [step[1][policy_id] for step in steps_in_last_epi]
            assert len(actions_played) == n_steps_in_last_epi
            actions_required_during_last_epi = actions_list[:len(all_steps)][-n_steps_in_last_epi:]
            for action_required, action_played in zip(actions_required_during_last_epi, actions_played):
                assert action_required == action_played

    assert_(rollout_length=20, num_episodes=1, actions_list=[0, 1] * 100)
    assert_(rollout_length=40, num_episodes=1, actions_list=[1, 1] * 100)
    assert_(rollout_length=77, num_episodes=2, actions_list=[0, 0] * 100)
    assert_(rollout_length=77, num_episodes=3, actions_list=[0, 1] * 100)
    assert_(rollout_length=6, num_episodes=3, actions_list=[1, 0] * 100)
