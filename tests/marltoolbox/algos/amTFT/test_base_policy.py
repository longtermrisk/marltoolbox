import copy

import random
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.sample_batch_builder import MultiAgentSampleBatchBuilder

from marltoolbox.algos import amTFT
from marltoolbox.algos.amTFT import base_policy
from marltoolbox.envs.matrix_sequential_social_dilemma import IteratedPrisonersDilemma
from marltoolbox.utils.postprocessing import WELFARE_UTILITARIAN


def init_amTFT(policy_config_update={}, policy_class=base_policy.amTFTPolicyBase):
    policy_config = copy.deepcopy(amTFT.DEFAULT_CONFIG)
    policy_config["welfare"] = WELFARE_UTILITARIAN
    policy_config.update(policy_config_update)
    env = IteratedPrisonersDilemma({})
    am_tft_policy = policy_class(env.OBSERVATION_SPACE, env.ACTION_SPACE, policy_config)
    return am_tft_policy, env


def test__select_witch_algo_to_use():
    am_tft_policy, env = init_amTFT()

    def assert_(working_state_idx, active_algo_idx):
        am_tft_policy.working_state = base_policy.WORKING_STATES[working_state_idx]
        am_tft_policy._select_witch_algo_to_use()
        assert am_tft_policy.active_algo_idx == active_algo_idx

    assert_(working_state_idx=0, active_algo_idx=base_policy.OWN_COOP_POLICY_IDX)
    assert_(working_state_idx=1, active_algo_idx=base_policy.OWN_SELFISH_POLICY_IDX)
    am_tft_policy.n_steps_to_punish = 0
    assert_(working_state_idx=2, active_algo_idx=base_policy.OWN_COOP_POLICY_IDX)
    am_tft_policy.n_steps_to_punish = 1
    assert_(working_state_idx=2, active_algo_idx=base_policy.OWN_SELFISH_POLICY_IDX)
    assert_(working_state_idx=3, active_algo_idx=base_policy.OWN_SELFISH_POLICY_IDX)
    assert_(working_state_idx=4, active_algo_idx=base_policy.OWN_COOP_POLICY_IDX)


def test___init__in_evaluation():
    am_tft_policy, env = init_amTFT({"working_state": base_policy.WORKING_STATES[2]})

    for algo in am_tft_policy.algorithms:
        assert not algo.model.training


def generate_fake_multiagent_batch(env, policies):
    multi_agent_batch_builder = MultiAgentSampleBatchBuilder(
        policy_map={player: policy for player, policy in zip(env.players_ids, policies)},
        clip_rewards=False,
        callbacks=DefaultCallbacks()
    )
    fake_actions = generate_fake_discrete_actions(env)
    env.reset()
    observations, rewards, done, info = env.step(fake_actions)
    for player_id in env.players_ids:
        step_player_values = {
            "eps_id": 0,
            "obs": observations[player_id],
            "new_obs": observations[player_id],
            "actions": fake_actions[player_id],
            "prev_actions": fake_actions[player_id],
            "rewards": rewards[player_id],
            "prev_rewards": rewards[player_id],
            "dones": done[player_id],
        }
        multi_agent_batch_builder.add_values(agent_id=player_id, policy_id=player_id, **step_player_values)
    multiagent_batch = multi_agent_batch_builder.build_and_reset()
    return multiagent_batch


def generate_fake_discrete_actions(env):
    return {player_id: random.randint(0, env.NUM_ACTIONS - 1) for player_id in env.players_ids}


def test_lr_update():
    base_lr = 2.2
    interm_global_timestep = 111
    final_lr = base_lr / 3.3
    am_tft_policy, env = init_amTFT({
        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": base_lr,
        # Learning rate schedule
        "lr_schedule": [(0, base_lr),
                        (interm_global_timestep, final_lr)],
    })

    multiagent_batch = generate_fake_multiagent_batch(env, policies=[am_tft_policy, am_tft_policy])
    one_policy_batch = multiagent_batch.policy_batches[env.players_ids[0]]

    am_tft_policy.on_global_var_update({"timestep": 0})
    am_tft_policy.learn_on_batch(one_policy_batch)
    for algo in am_tft_policy.algorithms:
        assert algo.cur_lr == base_lr
        for opt in algo._optimizers:
            for p in opt.param_groups:
                assert p["lr"] == algo.cur_lr

    am_tft_policy.on_global_var_update({"timestep": interm_global_timestep})
    am_tft_policy.learn_on_batch(one_policy_batch)
    for algo in am_tft_policy.algorithms:
        assert algo.cur_lr == final_lr
        for opt in algo._optimizers:
            for p in opt.param_groups:
                assert p["lr"] == algo.cur_lr


def test__is_punishment_planned():
    am_tft_policy, env = init_amTFT()
    am_tft_policy.n_steps_to_punish = 0
    assert not am_tft_policy._is_punishment_planned()
    am_tft_policy.n_steps_to_punish = 1
    assert am_tft_policy._is_punishment_planned()


def test_on_episode_end():
    am_tft_policy, env = init_amTFT({"working_state": base_policy.WORKING_STATES[2]})
    am_tft_policy.total_debit = 0
    am_tft_policy.n_steps_to_punish = 0
    am_tft_policy.on_episode_end()
    assert am_tft_policy.total_debit == 0
    assert am_tft_policy.n_steps_to_punish == 0
