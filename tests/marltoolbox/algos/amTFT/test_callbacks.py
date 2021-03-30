import os
import tempfile
import time
from unittest import mock
from unittest.mock import patch

import pytest
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR

from marltoolbox.algos import amTFT
from marltoolbox.envs.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma
from marltoolbox.examples.rllib_api.amtft_various_env import \
    get_rllib_config, get_hyperparameters
from marltoolbox.utils import postprocessing, log


@pytest.fixture()
def env_name():
    exp_name, _ = log.log_in_current_day_dir("testing")
    return exp_name


@pytest.fixture()
def logger_creator(env_name):
    logdir_prefix = env_name + '/'
    tail, head = os.path.split(env_name)
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
        if not os.path.exists(os.path.join(DEFAULT_RESULTS_DIR, env_name)):
            os.mkdir(os.path.join(DEFAULT_RESULTS_DIR, env_name))
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)
        return UnifiedLogger(config, logdir, loggers=None)

    return default_logger_creator


class PickableMock(mock.Mock):
    def __reduce__(self):
        return (mock.Mock, ())

mock.Mock = PickableMock

@pytest.fixture()
def dqn_trainer_wt_amtft_policies_in_ipd(
        logger_creator,
         ):

    train_n_replicates = 1
    debug = True
    hparams = get_hyperparameters(
        debug,
        train_n_replicates,
        filter_utilitarian=False,
        env="IteratedPrisonersDilemma")

    _, _, rllib_config = \
        get_rllib_config(
            hparams,
            welfare_fn=postprocessing.WELFARE_UTILITARIAN)

    rllib_config['env'] = IteratedPrisonersDilemma
    rllib_config['seed'] = int(time.time())

    policies = rllib_config["multiagent"]["policies"]
    for policy_id, policy_tuple in policies.items():
        policy_list = list(policy_tuple)
        policy_list[0] = amTFT.AmTFTRolloutsTorchPolicy
        policies[policy_id] = policy_list

    dqn_trainer = DQNTrainer(rllib_config,
                             logger_creator=logger_creator)
    return dqn_trainer


@pytest.fixture()
def rllib_worker_wt_amtft_policies_in_ipd(
        dqn_trainer_wt_amtft_policies_in_ipd):
    worker = dqn_trainer_wt_amtft_policies_in_ipd.workers._local_worker
    return worker

# TODO finish implementing it
# @patch(target='marltoolbox.algos.amTFT.AmTFTRolloutsTorchPolicy' \
#              '.on_episode_step',
#        # spec=True
#        )
# def test_on_episode_step_not_called_after_env_reset(
#         mock_on_episode_step,
#         rllib_worker_wt_amtft_policies_in_ipd):
#     worker = rllib_worker_wt_amtft_policies_in_ipd
#
#     am_tft_policy_row = worker.get_policy("player_row")
#     am_tft_policy_col = worker.get_policy("player_col")
#     print("am_tft_policy_row", am_tft_policy_row)
#     print("am_tft_policy_col", am_tft_policy_col)
#     print("worker.policy_map", worker.policy_map)
#
#     worker.input_reader.next()
#     # worker.input_reader.next()
#
#     for policy_id, policy in worker.policy_map.items():
#         print("policy_id", policy_id)
#         mock_on_episode_step.assert_called_once_with("file")
#         policy.assert_called_once_with('file')

