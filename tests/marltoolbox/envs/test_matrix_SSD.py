import numpy as np
import random

from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma, IteratedChicken, IteratedStagHunt, IteratedBoS


def init_env(max_steps, env_class, seed=None):
    config = {
        "max_steps": max_steps,
    }
    env = env_class(config)
    env.seed(seed)
    return env


def check_obs(obs, env):
    assert len(obs) == 2, "two players"
    for key, player_obs in obs.items():
        assert player_obs.shape == (env.NUM_STATES)
        assert player_obs.sum() == 1.0, f"one hot vector: {player_obs}"


def assert_logger_buffer_size_two_players(env, n_steps):
    assert len(env.cc_count) == n_steps
    assert len(env.dd_count) == n_steps
    assert len(env.cd_count) == n_steps
    assert len(env.dc_count) == n_steps


def test_reset():
    max_steps = 20
    env_class_all = [IteratedPrisonersDilemma, IteratedChicken, IteratedStagHunt, IteratedBoS]
    env_all = [ init_env(max_steps, env_class) for env_class in env_class_all]

    for env in env_all:
        obs = env.reset()
        check_obs(obs, env)
        assert_logger_buffer_size_two_players(env, n_steps=0)


def test_step():
    max_steps = 20
    env_class_all = [IteratedPrisonersDilemma, IteratedChicken, IteratedStagHunt, IteratedBoS]
    env_all = [ init_env(max_steps, env_class) for env_class in env_class_all]

    for env in env_all:
        obs = env.reset()
        check_obs(obs, env)
        assert_logger_buffer_size_two_players(env, n_steps=0)

        actions = {policy_id: random.randint(0, env.NUM_ACTIONS - 1) for policy_id in env.players_ids}
        obs, reward, done, info = env.step(actions)
        check_obs(obs, env)
        assert_logger_buffer_size_two_players(env, n_steps=1)
        assert not done["__all__"]


def test_multiple_steps():
    max_steps = 20
    env_class_all = [IteratedPrisonersDilemma, IteratedChicken, IteratedStagHunt, IteratedBoS]
    env_all = [ init_env(max_steps, env_class) for env_class in env_class_all]
    n_steps = int(max_steps * 0.75)

    for env in env_all:
        obs = env.reset()
        check_obs(obs, env)
        assert_logger_buffer_size_two_players(env, n_steps=0)

        for step_i in range(1, n_steps, 1):
            actions = {policy_id: random.randint(0, env.NUM_ACTIONS - 1) for policy_id in env.players_ids}
            obs, reward, done, info = env.step(actions)
            check_obs(obs, env)
            assert_logger_buffer_size_two_players(env, n_steps=step_i)
            assert not done["__all__"]


def test_multiple_episodes():
    max_steps = 20
    env_class_all = [IteratedPrisonersDilemma, IteratedChicken, IteratedStagHunt, IteratedBoS]
    env_all = [ init_env(max_steps, env_class) for env_class in env_class_all]
    n_steps = int(max_steps * 8.25)

    for env in env_all:
        obs = env.reset()
        check_obs(obs, env)
        assert_logger_buffer_size_two_players(env, n_steps=0)

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = {policy_id: [random.randint(0, env.NUM_ACTIONS - 1) for _ in range(batch_size)]
                       for policy_id in env.players_ids}
            obs, reward, done, info = env.step(actions)
            check_obs(obs, env)
            assert_logger_buffer_size_two_players(env, n_steps=step_i)
            assert not done["__all__"] or (step_i == max_steps and done["__all__"])
            if done["__all__"]:
                obs = env.reset()
                check_obs(obs, env)


def assert_info(n_steps, p_row_act, p_col_act, env, max_steps,
                CC, DD, CD, DC):
    step_i = 0
    for _ in range(n_steps):
        step_i += 1
        actions = {"player_row": p_row_act[step_i - 1],
                   "player_col": p_col_act[step_i - 1]}
        obs, reward, done, info = env.step(actions)
        check_obs(obs, env)
        assert_logger_buffer_size_two_players(env, n_steps=step_i)
        assert not done["__all__"] or (step_i == max_steps and done["__all__"])

        if done["__all__"]:
            assert info["player_row"]["CC"] == CC
            assert info["player_col"]["CC"] == CC
            assert info["player_row"]["DD"] == DD
            assert info["player_col"]["DD"] == DD
            assert info["player_row"]["CD"] == CD
            assert info["player_col"]["CD"] == CD
            assert info["player_row"]["DC"] == DC
            assert info["player_col"]["DC"] == DC

            obs = env.reset()
            check_obs(obs, env)
            assert_logger_buffer_size_two_players(env, n_steps=0)
            step_i = 0
#
#
# def test_logged_info_no_picking():
#     p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
#     p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     c_blue_pos = [None, None, None, None]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env in [coin_game, asymm_coin_game]:
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=0.0, blue_speed=0.0, red_own=None, blue_own=None)
#
#
# def test_logged_info__red_pick_red_all_the_time():
#     p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     c_blue_pos = [None, None, None, None]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=1.0, blue_speed=0.0, red_own=1.0, blue_own=None)
#
#
# def test_logged_info__blue_pick_red_all_the_time():
#     p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     c_blue_pos = [None, None, None, None]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=0.0)
#
#
# def test_logged_info__blue_pick_blue_all_the_time():
#     p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [None, None, None, None]
#     c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=1.0)
#
#
# def test_logged_info__red_pick_blue_all_the_time():
#     p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [None, None, None, None]
#     c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=1.0, blue_speed=0.0, red_own=0.0, blue_own=None)
#
#
# def test_logged_info__both_pick_blue_all_the_time():
#     p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [None, None, None, None]
#     c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=1.0, blue_speed=1.0, red_own=0.0, blue_own=1.0)
#
#
# def test_logged_info__both_pick_red_all_the_time():
#     p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     c_blue_pos = [None, None, None, None]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         print(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#               p_red_pos, p_blue_pos, c_red_pos, c_blue_pos)
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=1.0, blue_speed=1.0, red_own=1.0, blue_own=0.0)
#
#
# def test_logged_info__both_pick_red_half_the_time():
#     p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     c_blue_pos = [None, None, None, None]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=0.5, blue_speed=0.5, red_own=1.0, blue_own=0.0)
#
#
# def test_logged_info__both_pick_blue_half_the_time():
#     p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [None, None, None, None]
#     c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=0.5, blue_speed=0.5, red_own=0.0, blue_own=1.0)
#
#
# def test_logged_info__both_pick_blue():
#     p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [None, None, None, None]
#     c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=0.25, blue_speed=0.5, red_own=0.0, blue_own=1.0)
#
#
# def test_logged_info__pick_half_the_time_half_blue_half_red():
#     p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
#     p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
#     p_red_act = [0, 0, 0, 0]
#     p_blue_act = [0, 0, 0, 0]
#     c_red_pos = [[1, 1], None, [1, 1], None]
#     c_blue_pos = [None, [1, 1], None, [1, 1]]
#     max_steps, batch_size, grid_size = 4, 28, 3
#     n_steps = max_steps
#     coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)
#
#     for env_i, env in enumerate([coin_game, asymm_coin_game]):
#         obs = env.reset()
#         check_obs(obs, batch_size, grid_size)
#         assert_logger_buffer_size(env, n_steps=0)
#         overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
#
#         assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
#                     p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
#                     red_speed=0.5, blue_speed=0.5, red_own=0.5, blue_own=0.5)
