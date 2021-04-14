import os

import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer

from marltoolbox.examples.rllib_api.pg_ipd import get_rllib_config
from marltoolbox.utils import log, miscellaneous, restore
from marltoolbox.utils import self_and_cross_perf
from marltoolbox.utils.miscellaneous import get_random_seeds


def _init_evaluator():
    exp_name, _ = log.log_in_current_day_dir("testing")

    rllib_config, stop_config = get_rllib_config(seeds=get_random_seeds(1))

    evaluator = self_and_cross_perf.SelfAndCrossPlayEvaluator(
        exp_name=exp_name,
    )
    evaluator.define_the_experiment_to_run(
        evaluation_config=rllib_config,
        stop_config=stop_config,
        TrainerClass=PGTrainer,
    )

    return evaluator


def _train_pg_in_ipd(train_n_replicates):
    debug = True
    stop_iters = 200
    tf = False
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("testing")

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)

    rllib_config, stop_config = get_rllib_config(seeds, debug, stop_iters, tf)
    tune_analysis = tune.run(
        PGTrainer,
        config=rllib_config,
        stop=stop_config,
        checkpoint_freq=0,
        checkpoint_at_end=True,
        name=exp_name,
        metric="episode_reward_mean",
        mode="max",
    )
    ray.shutdown()
    return tune_analysis, seeds


def _load_tune_analysis(evaluator, train_n_replicates, exp_name):
    tune_analysis, seeds = _train_pg_in_ipd(train_n_replicates)
    tune_results = {exp_name: tune_analysis}
    evaluator.preload_checkpoints_from_tune_results(tune_results)

    return seeds


def test__config_is_for_two_players():
    evaluator = _init_evaluator()
    evaluator._config_is_for_two_players()


def test__is_policy_to_load():
    evaluator = _init_evaluator()
    assert evaluator._is_policy_to_load(
        policy_id="Ok", policies_to_load_from_checkpoint=["All"]
    )
    assert evaluator._is_policy_to_load(
        policy_id="Ok", policies_to_load_from_checkpoint=["Ok"]
    )
    assert not evaluator._is_policy_to_load(
        policy_id="NotOk", policies_to_load_from_checkpoint=["Ok"]
    )
    assert not evaluator._is_policy_to_load(
        policy_id="NotOk", policies_to_load_from_checkpoint=[]
    )


def test__extract_groups_of_checkpoints():
    evaluator = _init_evaluator()

    def assert_(exp_name, train_n_replicates):
        seeds = _load_tune_analysis(evaluator, train_n_replicates, exp_name)
        assert len(evaluator.checkpoints) == train_n_replicates
        for idx, checkpoint in enumerate(evaluator.checkpoints):
            assert str(seeds[idx]) in checkpoint["path"]
            assert checkpoint["group_name"] == exp_name

    assert_("", 2)
    assert_("random_exp", 3)


def test__get_opponents_per_checkpoints():
    evaluator = _init_evaluator()
    exp_name, train_n_replicates = "", 3
    _load_tune_analysis(evaluator, train_n_replicates, exp_name)
    n_cross_play_per_checkpoint = train_n_replicates - 1
    opponents_per_checkpoint = evaluator._get_opponents_per_checkpoints(
        n_cross_play_per_checkpoint
    )
    for checkpoint_idx, opponents in enumerate(opponents_per_checkpoint):
        for opp_idx in opponents:
            assert opp_idx != checkpoint_idx


def test__produce_config_variations():
    # A bit useless this test

    evaluator = _init_evaluator()
    exp_name, train_n_replicates = "", 4
    _load_tune_analysis(evaluator, train_n_replicates, exp_name)

    def assert_(n_same_play_per_checkpoint, n_cross_play_per_checkpoint):
        opponents_per_checkpoint = evaluator._get_opponents_per_checkpoints(
            n_cross_play_per_checkpoint
        )
        (
            all_config_variations,
            all_metadata,
        ) = evaluator._produce_config_variations(
            n_same_play_per_checkpoint,
            n_cross_play_per_checkpoint,
            opponents_per_checkpoint,
        )

        assert (
            len(all_config_variations)
            == (n_cross_play_per_checkpoint + n_same_play_per_checkpoint)
            * train_n_replicates
        )
        assert (
            len(all_metadata)
            == (n_cross_play_per_checkpoint + n_same_play_per_checkpoint)
            * train_n_replicates
        )

    assert_(1, 1)
    assert_(2, 3)
    assert_(7, 3)


def test__prepare_one_master_config_dict():
    evaluator = _init_evaluator()
    exp_name, train_n_replicates = "", 4
    _load_tune_analysis(evaluator, train_n_replicates, exp_name)

    def assert_(n_same_play_per_checkpoint, n_cross_play_per_checkpoint):
        (
            master_config,
            all_metadata,
        ) = evaluator._prepare_one_master_config_dict(
            n_same_play_per_checkpoint, n_cross_play_per_checkpoint
        )

        assert (
            len(master_config["multiagent"]["policies"]["grid_search"])
            == (n_cross_play_per_checkpoint + n_same_play_per_checkpoint)
            * train_n_replicates
        )

    assert_(1, 0)
    assert_(2, 3)
    assert_(7, 3)


def test__get_config_for_one_same_play():
    evaluator = _init_evaluator()
    exp_name, train_n_replicates = "", 4
    _load_tune_analysis(evaluator, train_n_replicates, exp_name)

    def assert_(checkpoint_i):
        metadata, config_copy = evaluator._get_config_for_one_self_play(
            checkpoint_i
        )

        own_policy_id = evaluator.policies_to_load_from_checkpoint[0]
        assert (
            metadata[own_policy_id]["checkpoint_path"]
            == evaluator.checkpoints[checkpoint_i]["path"]
        )
        assert metadata[own_policy_id]["checkpoint_i"] == checkpoint_i
        assert (
            config_copy["multiagent"]["policies"][own_policy_id][3][
                restore.LOAD_FROM_CONFIG_KEY
            ][0]
            == evaluator.checkpoints[checkpoint_i]["path"]
        )

        opp_policy_id = evaluator.policies_to_load_from_checkpoint[1]
        assert (
            metadata[opp_policy_id]["checkpoint_path"]
            == evaluator.checkpoints[checkpoint_i]["path"]
        )
        assert metadata[opp_policy_id]["checkpoint_i"] == checkpoint_i
        assert (
            config_copy["multiagent"]["policies"][opp_policy_id][3][
                restore.LOAD_FROM_CONFIG_KEY
            ][0]
            == evaluator.checkpoints[checkpoint_i]["path"]
        )

    for i in range(train_n_replicates):
        assert_(i)


def test__get_config_for_one_cross_play():
    evaluator = _init_evaluator()
    exp_name, train_n_replicates = "", 4
    _load_tune_analysis(evaluator, train_n_replicates, exp_name)

    def assert_(checkpoint_i):
        n_cross_play_per_checkpoint = train_n_replicates - 1
        opponents_idx_per_checkpoint = (
            evaluator._get_opponents_per_checkpoints(
                n_cross_play_per_checkpoint
            )
        )
        opponents_idx = opponents_idx_per_checkpoint[checkpoint_i]
        for cross_play_i in range(n_cross_play_per_checkpoint):
            metadata, config_copy = evaluator._get_config_for_one_cross_play(
                checkpoint_i, opponent_i=opponents_idx[cross_play_i]
            )

            own_policy_id = evaluator.policies_to_load_from_checkpoint[0]
            assert (
                metadata[own_policy_id]["checkpoint_path"]
                == evaluator.checkpoints[checkpoint_i]["path"]
            )
            assert metadata[own_policy_id]["checkpoint_i"] == checkpoint_i
            assert (
                config_copy["multiagent"]["policies"][own_policy_id][3][
                    restore.LOAD_FROM_CONFIG_KEY
                ][0]
                == evaluator.checkpoints[checkpoint_i]["path"]
            )

            opponent_checkpoint = evaluator.checkpoints[
                opponents_idx[cross_play_i]
            ]
            opp_policy_id = evaluator.policies_to_load_from_checkpoint[1]
            assert (
                metadata[opp_policy_id]["checkpoint_path"]
                == opponent_checkpoint["path"]
            )
            assert (
                metadata[opp_policy_id]["checkpoint_i"]
                == opponents_idx[cross_play_i]
            )
            assert (
                config_copy["multiagent"]["policies"][opp_policy_id][3][
                    restore.LOAD_FROM_CONFIG_KEY
                ][0]
                == opponent_checkpoint["path"]
            )

    for i in range(train_n_replicates):
        assert_(i)
