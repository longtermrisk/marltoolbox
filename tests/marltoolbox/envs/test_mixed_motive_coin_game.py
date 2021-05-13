import random

import numpy as np

from marltoolbox.envs.mixed_motive_coin_game import MixedMotiveCoinGame
from coin_game_tests_utils import (
    check_custom_obs,
    assert_logger_buffer_size,
    helper_test_reset,
    helper_test_step,
    init_several_envs,
    helper_test_multiple_steps,
    helper_test_multi_ple_episodes,
    helper_assert_info,
)


# TODO add tests for grid_size != 3


def init_my_envs(
    max_steps,
    grid_size,
    players_can_pick_same_coin=True,
    same_obs_for_each_player=True,
):
    return init_several_envs(
        classes=(MixedMotiveCoinGame,),
        max_steps=max_steps,
        grid_size=grid_size,
        players_can_pick_same_coin=players_can_pick_same_coin,
        same_obs_for_each_player=same_obs_for_each_player,
    )


def check_obs(obs, grid_size):
    check_custom_obs(obs, grid_size, n_in_2_and_above=2.0)


def test_reset():
    max_steps, grid_size = 20, 3
    envs = init_my_envs(max_steps, grid_size)
    helper_test_reset(envs, check_obs, grid_size=grid_size)


def test_step():
    max_steps, grid_size = 20, 3
    envs = init_my_envs(max_steps, grid_size)
    helper_test_step(envs, check_obs, grid_size=grid_size)


def test_multiple_steps():
    max_steps, grid_size = 20, 3
    n_steps = int(max_steps * 0.75)
    envs = init_my_envs(max_steps, grid_size)
    helper_test_multiple_steps(
        envs,
        n_steps,
        check_obs,
        grid_size=grid_size,
    )


def test_multiple_episodes():
    max_steps, grid_size = 20, 3
    n_steps = int(max_steps * 8.25)
    envs = init_my_envs(max_steps, grid_size)
    helper_test_multi_ple_episodes(
        envs,
        n_steps,
        max_steps,
        check_obs,
        grid_size=grid_size,
    )


def overwrite_pos(env, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, **kwargs):
    env.red_pos = p_red_pos
    env.blue_pos = p_blue_pos
    env.red_coin_pos = c_red_pos
    env.blue_coin_pos = c_blue_pos

    env.red_pos = np.array(env.red_pos)
    env.blue_pos = np.array(env.blue_pos)
    env.red_coin_pos = np.array(env.red_coin_pos)
    env.blue_coin_pos = np.array(env.blue_coin_pos)


def test_logged_info_no_picking():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    for env in envs:
        obs = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(
            env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0]
        )

        helper_assert_info(
            n_steps=n_steps,
            p_red_act=p_red_act,
            p_blue_act=p_blue_act,
            envs=envs,
            grid_size=grid_size,
            max_steps=max_steps,
            p_red_pos=p_red_pos,
            p_blue_pos=p_blue_pos,
            c_red_pos=c_red_pos,
            c_blue_pos=c_blue_pos,
            check_obs_fn=check_obs,
            overwrite_pos_fn=overwrite_pos,
            red_speed=0.0,
            blue_speed=0.0,
            red_own=None,
            blue_own=None,
        )


def test_logged_info__red_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_logged_info__blue_pick_red_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_logged_info__blue_pick_blue_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_logged_info__red_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_logged_info__both_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=1.0,
        blue_speed=1.0,
        red_own=0.0,
        blue_own=1.0,
    )


def test_logged_info__both_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=1.0,
        blue_speed=1.0,
        red_own=1.0,
        blue_own=0.0,
    )


def test_logged_info__both_pick_red_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_logged_info__both_pick_blue_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_logged_info__both_pick_blue():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_logged_info__pick_half_the_time_half_blue_half_red():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [0, 0], [1, 1], [0, 0]]
    c_blue_pos = [[0, 0], [1, 1], [0, 0], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
    )


def test_observations_are_invariant_to_the_player_trained_in_reset():
    p_red_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [2, 0],
        [0, 1],
        [2, 2],
        [1, 2],
    ]
    p_blue_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 0],
        [1, 2],
        [2, 2],
    ]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [
        [1, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
    ]
    c_blue_pos = [
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
        [0, 0],
    ]
    max_steps, grid_size = 10, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, same_obs_for_each_player=False)

    for env_i, env in enumerate(envs):
        obs = env.reset()
        assert_obs_is_symmetrical(obs, env)
        step_i = 0
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            c_red_pos[step_i],
            c_blue_pos[step_i],
        )

        for _ in range(n_steps):
            step_i += 1
            actions = {
                "player_red": p_red_act[step_i - 1],
                "player_blue": p_blue_act[step_i - 1],
            }
            _, _, _, _ = env.step(actions)

            if step_i == max_steps:
                break

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                c_red_pos[step_i],
                c_blue_pos[step_i],
            )


def assert_obs_is_symmetrical(obs, env):
    assert np.all(
        obs[env.players_ids[0]][..., 0] == obs[env.players_ids[1]][..., 1]
    )
    assert np.all(
        obs[env.players_ids[1]][..., 0] == obs[env.players_ids[0]][..., 1]
    )
    assert np.all(
        obs[env.players_ids[0]][..., 2] == obs[env.players_ids[1]][..., 3]
    )
    assert np.all(
        obs[env.players_ids[1]][..., 2] == obs[env.players_ids[0]][..., 3]
    )


def test_observations_are_invariant_to_the_player_trained_in_step():
    p_red_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [2, 0],
        [0, 1],
        [2, 2],
        [1, 2],
    ]
    p_blue_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 0],
        [1, 2],
        [2, 2],
    ]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [
        [1, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
    ]
    c_blue_pos = [
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
        [0, 0],
    ]
    max_steps, grid_size = 10, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, same_obs_for_each_player=False)

    for env_i, env in enumerate(envs):
        _ = env.reset()
        step_i = 0
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            c_red_pos[step_i],
            c_blue_pos[step_i],
        )

        for _ in range(n_steps):
            step_i += 1
            actions = {
                "player_red": p_red_act[step_i - 1],
                "player_blue": p_blue_act[step_i - 1],
            }
            obs, reward, done, info = env.step(actions)

            # assert observations are symmetrical respective to the actions
            if step_i % 2 == 1:
                obs_step_odd = obs
            elif step_i % 2 == 0:
                assert np.all(
                    obs[env.players_ids[0]] == obs_step_odd[env.players_ids[1]]
                )
                assert np.all(
                    obs[env.players_ids[1]] == obs_step_odd[env.players_ids[0]]
                )
            assert_obs_is_symmetrical(obs, env)

            if step_i == max_steps:
                break

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                c_red_pos[step_i],
                c_blue_pos[step_i],
            )


def test_observations_are_not_invariant_to_the_player_trained_in_reset():
    p_red_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [2, 0],
        [0, 1],
        [2, 2],
        [1, 2],
    ]
    p_blue_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 0],
        [1, 2],
        [2, 2],
    ]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [
        [1, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
    ]
    c_blue_pos = [
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
        [0, 0],
    ]
    max_steps, grid_size = 10, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, same_obs_for_each_player=True)

    for env_i, env in enumerate(envs):
        obs = env.reset()
        assert_obs_is_not_symmetrical(obs, env)
        step_i = 0
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            c_red_pos[step_i],
            c_blue_pos[step_i],
        )

        for _ in range(n_steps):
            step_i += 1
            actions = {
                "player_red": p_red_act[step_i - 1],
                "player_blue": p_blue_act[step_i - 1],
            }
            _, _, _, _ = env.step(actions)

            if step_i == max_steps:
                break

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                c_red_pos[step_i],
                c_blue_pos[step_i],
            )


def assert_obs_is_not_symmetrical(obs, env):
    assert np.all(obs[env.players_ids[0]] == obs[env.players_ids[1]])


def test_observations_are_not_invariant_to_the_player_trained_in_step():
    p_red_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [2, 0],
        [0, 1],
        [2, 2],
        [1, 2],
    ]
    p_blue_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 0],
        [1, 2],
        [2, 2],
    ]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [
        [1, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
    ]
    c_blue_pos = [
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
        [0, 0],
    ]
    max_steps, grid_size = 10, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, same_obs_for_each_player=True)

    for env_i, env in enumerate(envs):
        _ = env.reset()
        step_i = 0
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            c_red_pos[step_i],
            c_blue_pos[step_i],
        )

        for _ in range(n_steps):
            step_i += 1
            actions = {
                "player_red": p_red_act[step_i - 1],
                "player_blue": p_blue_act[step_i - 1],
            }
            obs, reward, done, info = env.step(actions)

            # assert observations are symmetrical respective to the actions
            if step_i % 2 == 1:
                obs_step_odd = obs
            elif step_i % 2 == 0:
                assert np.any(
                    obs[env.players_ids[0]] != obs_step_odd[env.players_ids[1]]
                )
                assert np.any(
                    obs[env.players_ids[1]] != obs_step_odd[env.players_ids[0]]
                )
            assert_obs_is_not_symmetrical(obs, env)

            if step_i == max_steps:
                break

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                c_red_pos[step_i],
                c_blue_pos[step_i],
            )
