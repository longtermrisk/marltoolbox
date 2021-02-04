import copy
import random

import numpy as np

from marltoolbox.envs.vectorized_coin_game import CoinGame, AsymCoinGame


# TODO add tests for grid_size != 3
# TODO add tests for position in episode in 5th

def init_env(max_steps, batch_size, env_class, seed=None, grid_size=3):
    config = {
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grid_size": grid_size,
    }
    env = env_class(config)
    env.seed(seed)
    return env


def check_obs(obs, batch_size, grid_size):
    assert len(obs) == 2, "two players"
    for i in range(batch_size):
        for key, player_obs in obs.items():
            assert player_obs.shape == (batch_size, grid_size, grid_size, 4)
            assert player_obs[i, ..., 0].sum() == 1.0, f"observe 1 player red in grid: {player_obs[i, ..., 0]}"
            assert player_obs[i, ..., 1].sum() == 1.0, f"observe 1 player blue in grid: {player_obs[i, ..., 1]}"
            assert player_obs[i, ..., 2:].sum() == 1.0, f"observe 1 coin in grid: {player_obs[i, ..., 0]}"


def assert_logger_buffer_size(env, n_steps):
    assert len(env.red_pick) == n_steps
    assert len(env.red_pick_own) == n_steps
    assert len(env.blue_pick) == n_steps
    assert len(env.blue_pick_own) == n_steps


def test_reset():
    max_steps, batch_size, grid_size = 20, 5, 3
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env in [coin_game, asymm_coin_game]:
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)


def test_step():
    max_steps, batch_size, grid_size = 20, 5, 3
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env in [coin_game, asymm_coin_game]:
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)

        actions = {policy_id: [random.randint(0, env.NUM_ACTIONS - 1) for _ in range(batch_size)]
                   for policy_id in env.players_ids}
        obs, reward, done, info = env.step(actions)
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=1)
        assert not done["__all__"]


def test_multiple_steps():
    max_steps, batch_size, grid_size = 20, 5, 3
    n_steps = int(max_steps * 0.75)
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env in [coin_game, asymm_coin_game]:
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)

        for step_i in range(1, n_steps, 1):
            actions = {policy_id: [random.randint(0, env.NUM_ACTIONS - 1) for _ in range(batch_size)]
                       for policy_id in env.players_ids}
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done["__all__"]


def test_multiple_episodes():
    max_steps, batch_size, grid_size = 20, 100, 3
    n_steps = int(max_steps * 8.25)
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env in [coin_game, asymm_coin_game]:
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = {policy_id: [random.randint(0, env.NUM_ACTIONS - 1) for _ in range(batch_size)]
                       for policy_id in env.players_ids}
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done["__all__"] or (step_i == max_steps and done["__all__"])
            if done["__all__"]:
                obs = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0


def overwrite_pos(batch_size, env, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos):
    assert c_red_pos is None or c_blue_pos is None
    if c_red_pos is None:
        env.red_coin = [0] * batch_size
        coin_pos = c_blue_pos
    if c_blue_pos is None:
        env.red_coin = [1] * batch_size
        coin_pos = c_red_pos

    env.red_pos = [p_red_pos] * batch_size
    env.blue_pos = [p_blue_pos] * batch_size
    env.coin_pos = [coin_pos] * batch_size

    env.red_pos = np.array(env.red_pos)
    env.blue_pos = np.array(env.blue_pos)
    env.coin_pos = np.array(env.coin_pos)
    env.red_coin = np.array(env.red_coin)


def assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                red_speed, blue_speed, red_own, blue_own):
    step_i = 0
    for _ in range(n_steps):
        step_i += 1
        actions = {"player_red": [p_red_act[step_i - 1]] * batch_size,
                   "player_blue": [p_blue_act[step_i - 1]] * batch_size}
        obs, reward, done, info = env.step(actions)
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=step_i)
        assert not done["__all__"] or (step_i == max_steps and done["__all__"])

        if done["__all__"]:
            assert info["player_red"]["pick_speed"] == red_speed
            assert info["player_blue"]["pick_speed"] == blue_speed

            if red_own is None:
                assert "pick_own_color" not in info["player_red"]
            else:
                assert info["player_red"]["pick_own_color"] == red_own
            if blue_own is None:
                assert "pick_own_color" not in info["player_blue"]
            else:
                assert info["player_blue"]["pick_own_color"] == blue_own

            obs = env.reset()
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=0)
            step_i = 0

        overwrite_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                      c_blue_pos[step_i])


def test_logged_info_no_picking():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env in [coin_game, asymm_coin_game]:
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=0.0, blue_speed=0.0, red_own=None, blue_own=None)


def test_logged_info__red_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=1.0, blue_speed=0.0, red_own=1.0, blue_own=None)


def test_logged_info__blue_pick_red_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=0.0)


def test_logged_info__blue_pick_blue_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=1.0)


def test_logged_info__red_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=1.0, blue_speed=0.0, red_own=0.0, blue_own=None)


def test_logged_info__both_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=1.0, blue_speed=1.0, red_own=0.0, blue_own=1.0)


def test_logged_info__both_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        print(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
              p_red_pos, p_blue_pos, c_red_pos, c_blue_pos)
        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=1.0, blue_speed=1.0, red_own=1.0, blue_own=0.0)


def test_logged_info__both_pick_red_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=0.5, blue_speed=0.5, red_own=1.0, blue_own=0.0)


def test_logged_info__both_pick_blue_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=0.5, blue_speed=0.5, red_own=0.0, blue_own=1.0)


def test_logged_info__both_pick_blue():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=0.25, blue_speed=0.5, red_own=0.0, blue_own=1.0)


def test_logged_info__pick_half_the_time_half_blue_half_red():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], None, [1, 1], None]
    c_blue_pos = [None, [1, 1], None, [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        assert_info(n_steps, batch_size, p_red_act, p_blue_act, env, grid_size, max_steps,
                    p_red_pos, p_blue_pos, c_red_pos, c_blue_pos,
                    red_speed=0.5, blue_speed=0.5, red_own=0.5, blue_own=0.5)


def test_get_and_set_env_state():
    max_steps, batch_size, grid_size = 20, 100, 3
    n_steps = int(max_steps * 8.25)
    coin_game = init_env(max_steps, batch_size, CoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCoinGame, grid_size)

    for env in [coin_game, asymm_coin_game]:
        obs = env.reset()
        initial_env_state = env._get_env_state()
        initial_env_state_saved = copy.deepcopy(initial_env_state)
        env_initial = copy.deepcopy(env)

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = {policy_id: [random.randint(0, env.NUM_ACTIONS - 1) for _ in range(batch_size)]
                       for policy_id in env.players_ids}
            obs, reward, done, info = env.step(actions)

            assert all([v == initial_env_state_saved[k]
                        if not isinstance(v, np.ndarray)
                        else (v == initial_env_state_saved[k]).all()
                        for k, v in initial_env_state.items()])
            env_state_after_step = env._get_env_state()
            env_after_step = copy.deepcopy(env)

            env._set_env_state(initial_env_state)
            env_vars, env_initial_vars = vars(env), vars(env_initial)
            env_vars.pop("np_random", None)
            env_initial_vars.pop("np_random", None)
            assert all([v == env_initial_vars[k]
                        if not isinstance(v, np.ndarray)
                        else (v == env_initial_vars[k]).all()
                        for k, v in env_vars.items()])

            env._set_env_state(env_state_after_step)
            env_vars, env_after_step_vars = vars(env), vars(env_after_step)
            env_vars.pop("np_random", None)
            env_after_step_vars.pop("np_random", None)
            assert all([v == env_after_step_vars[k]
                        if not isinstance(v, np.ndarray)
                        else (v == env_after_step_vars[k]).all()
                        for k, v in env_vars.items()])

            if done["__all__"]:
                obs = env.reset()
                step_i = 0
