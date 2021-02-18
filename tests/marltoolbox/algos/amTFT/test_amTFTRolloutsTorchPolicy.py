from marltoolbox.algos import amTFT
from marltoolbox.algos.amTFT import base_policy
from test_base_policy import init_amTFT, generate_fake_discrete_actions


def test_compute_actions_overwrite():
    am_tft_policy, env = init_amTFT(policy_class=amTFT.amTFTRolloutsTorchPolicy)

    fake_actions = generate_fake_discrete_actions(env)
    env.reset()
    observations, rewards, done, info = env.step(fake_actions)

    am_tft_policy.use_opponent_policies = True
    fake_actions, fake_state_out, fake_extra_fetches = "fake", "fake", "fake"
    fake_actions_2nd, fake_state_out_2nd, fake_extra_fetches_2nd = "fake_2nd", "fake_2nd", "fake_2nd"
    am_tft_policy.overwrite_action = [(fake_actions, fake_state_out, fake_extra_fetches),
                                      (fake_actions_2nd, fake_state_out_2nd, fake_extra_fetches_2nd)]
    actions, state_out, extra_fetches = am_tft_policy.compute_actions(observations[env.players_ids[0]])
    assert actions == fake_actions
    assert state_out == fake_state_out
    assert extra_fetches == fake_extra_fetches
    actions, state_out, extra_fetches = am_tft_policy.compute_actions(observations[env.players_ids[0]])
    assert actions == fake_actions_2nd
    assert state_out == fake_state_out_2nd
    assert extra_fetches == fake_extra_fetches_2nd


def test__select_algo_to_use_in_eval():
    am_tft_policy, env = init_amTFT(policy_class=amTFT.amTFTRolloutsTorchPolicy)

    def assert_(working_state_idx, active_algo_idx):
        am_tft_policy.working_state = base_policy.WORKING_STATES[working_state_idx]
        am_tft_policy._select_witch_algo_to_use()
        assert am_tft_policy.active_algo_idx == active_algo_idx

    am_tft_policy.use_opponent_policies = False
    am_tft_policy.n_steps_to_punish = 0
    assert_(working_state_idx=2, active_algo_idx=base_policy.OWN_COOP_POLICY_IDX)
    am_tft_policy.use_opponent_policies = False
    am_tft_policy.n_steps_to_punish = 1
    assert_(working_state_idx=2, active_algo_idx=base_policy.OWN_SELFISH_POLICY_IDX)

    am_tft_policy.use_opponent_policies = True
    am_tft_policy.performing_rollouts = True
    am_tft_policy.n_steps_to_punish_opponent = 0
    assert_(working_state_idx=2, active_algo_idx=base_policy.OPP_COOP_POLICY_IDX)
    am_tft_policy.use_opponent_policies = True
    am_tft_policy.performing_rollouts = True
    am_tft_policy.n_steps_to_punish_opponent = 1
    assert_(working_state_idx=2, active_algo_idx=base_policy.OPP_SELFISH_POLICY_IDX)
