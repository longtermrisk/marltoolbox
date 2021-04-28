import copy
import random

import numpy as np

from marltoolbox.envs.vectorized_mixed_motive_coin_game import (
    VectMixedMotiveCG,
)
from test_coin_game import (
    assert_obs_is_symmetrical,
    assert_obs_is_not_symmetrical,
)
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
    batch_size,
    grid_size,
    players_can_pick_same_coin=True,
    same_obs_for_each_player=True,
):
    return init_several_envs(
        classes=(VectMixedMotiveCG,),
        max_steps=max_steps,
        grid_size=grid_size,
        batch_size=batch_size,
        players_can_pick_same_coin=players_can_pick_same_coin,
        same_obs_for_each_player=same_obs_for_each_player,
    )


def check_obs(obs, batch_size, grid_size):
    check_custom_obs(
        obs, grid_size, n_in_2_and_above=2.0, batch_size=batch_size
    )


def test_reset():
    max_steps, batch_size, grid_size = 20, 5, 3
    envs = init_my_envs(max_steps, batch_size, grid_size)
    helper_test_reset(
        envs, check_obs, grid_size=grid_size, batch_size=batch_size
    )


def test_step():
    max_steps, batch_size, grid_size = 20, 5, 3
    envs = init_my_envs(max_steps, batch_size, grid_size)
    helper_test_step(
        envs,
        check_obs,
        grid_size=grid_size,
        batch_size=batch_size,
    )


def test_multiple_steps():
    max_steps, batch_size, grid_size = 20, 5, 3
    n_steps = int(max_steps * 0.75)
    envs = init_my_envs(max_steps, batch_size, grid_size)
    helper_test_multiple_steps(
        envs,
        n_steps,
        check_obs,
        grid_size=grid_size,
        batch_size=batch_size,
    )


def test_multiple_episodes():
    max_steps, batch_size, grid_size = 20, 100, 3
    n_steps = int(max_steps * 8.25)
    envs = init_my_envs(max_steps, batch_size, grid_size)
    helper_test_multi_ple_episodes(
        envs,
        n_steps,
        max_steps,
        check_obs,
        grid_size=grid_size,
        batch_size=batch_size,
    )


def overwrite_pos(
    step_i,
    batch_deltas,
    n_steps_in_epi,
    env,
    p_red_pos,
    p_blue_pos,
    c_red_pos,
    c_blue_pos,
):
    assert len(p_red_pos) == n_steps_in_epi
    assert len(p_blue_pos) == n_steps_in_epi
    assert len(c_red_pos) == n_steps_in_epi
    assert len(c_blue_pos) == n_steps_in_epi

    env.red_pos = [
        p_red_pos[(step_i + delta) % n_steps_in_epi] for delta in batch_deltas
    ]
    env.blue_pos = [
        p_blue_pos[(step_i + delta) % n_steps_in_epi] for delta in batch_deltas
    ]
    env.red_coin_pos = [
        c_red_pos[(step_i + delta) % n_steps_in_epi] for delta in batch_deltas
    ]
    env.blue_coin_pos = [
        c_blue_pos[(step_i + delta) % n_steps_in_epi] for delta in batch_deltas
    ]

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
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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


def test_logged_info__red_pick_blue_all_the_time_wt_difference_in_actions():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 1], [0, 1]]
    p_red_act = [0, 1, 2, 3]
    p_blue_act = [0, 1, 2, 3]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 2], [2, 0], [0, 0]]
    max_steps, batch_size, grid_size = 4, 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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


def test_logged_info__pick_slowly_red_coin():
    p_red_pos = [[1, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
        red_speed=0.25,
        blue_speed=0.25,
        red_own=1.0,
        blue_own=0.0,
    )


def test_logged_info__pick_slowly_blue_coin():
    p_red_pos = [[1, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
        red_speed=0.25,
        blue_speed=0.25,
        red_own=0.0,
        blue_own=1.0,
    )


def test_logged_info__pick_quickly_red_coin():
    p_red_pos = [[1, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [0, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
        red_speed=0.75,
        blue_speed=0.75,
        red_own=1.0,
        blue_own=0.0,
    )


def test_logged_info__pick_quickly_blue_coin():
    p_red_pos = [[1, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [0, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
        red_speed=0.75,
        blue_speed=0.75,
        red_own=0.0,
        blue_own=1.0,
    )


def test_logged_info__pick_slowly_mixed_coin():
    p_red_pos = [[1, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [0, 0], [0, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [0, 0], [0, 0]]
    c_blue_pos = [[0, 0], [0, 0], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
        red_speed=0.50,
        blue_speed=0.50,
        red_own=0.5,
        blue_own=0.5,
    )


def test_logged_info__pick_quickly_mixed_coin():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [0, 0], [0, 0]]
    c_blue_pos = [[0, 0], [0, 0], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, batch_size, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        batch_size=batch_size,
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
        red_own=0.5,
        blue_own=0.5,
    )


def test_get_and_set_env_state():
    max_steps, batch_size, grid_size = 20, 100, 3
    n_steps = int(max_steps * 8.25)
    envs = init_my_envs(max_steps, batch_size, grid_size)

    for env in envs:
        obs = env.reset()
        initial_env_state = env._save_env()
        initial_env_state_saved = copy.deepcopy(initial_env_state)
        env_initial = copy.deepcopy(env)

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = {
                policy_id: [
                    random.randint(0, env.NUM_ACTIONS - 1)
                    for _ in range(batch_size)
                ]
                for policy_id in env.players_ids
            }
            obs, reward, done, info = env.step(actions)

            assert all(
                [
                    v == initial_env_state_saved[k]
                    if not isinstance(v, np.ndarray)
                    else (v == initial_env_state_saved[k]).all()
                    for k, v in initial_env_state.items()
                ]
            )
            env_state_after_step = env._save_env()
            env_after_step = copy.deepcopy(env)

            env._load_env(initial_env_state)
            env_vars, env_initial_vars = vars(env), vars(env_initial)
            env_vars.pop("np_random", None)
            env_initial_vars.pop("np_random", None)
            assert all(
                [
                    v == env_initial_vars[k]
                    if not isinstance(v, np.ndarray)
                    else (v == env_initial_vars[k]).all()
                    for k, v in env_vars.items()
                ]
            )

            env._load_env(env_state_after_step)
            env_vars, env_after_step_vars = vars(env), vars(env_after_step)
            env_vars.pop("np_random", None)
            env_after_step_vars.pop("np_random", None)
            assert all(
                [
                    v == env_after_step_vars[k]
                    if not isinstance(v, np.ndarray)
                    else (v == env_after_step_vars[k]).all()
                    for k, v in env_vars.items()
                ]
            )

            if done["__all__"]:
                obs = env.reset()
                step_i = 0


def test_observations_are_invariant_to_the_player_trained_wt_step():
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
    max_steps, batch_size, grid_size = 10, 52, 3
    n_steps = max_steps
    envs = init_my_envs(
        max_steps, batch_size, grid_size, same_obs_for_each_player=False
    )

    batch_deltas = [
        i % max_steps if i % 2 == 0 else i % max_steps - 1
        for i in range(batch_size)
    ]

    for env_i, env in enumerate(envs):
        _ = env.reset()
        step_i = 0

        for _ in range(n_steps):
            overwrite_pos(
                step_i,
                batch_deltas,
                max_steps,
                env,
                p_red_pos,
                p_blue_pos,
                c_red_pos,
                c_blue_pos,
            )
            actions = {
                "player_red": [
                    p_red_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
                "player_blue": [
                    p_blue_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
            }
            obs, reward, done, info = env.step(actions)

            step_i += 1
            # assert that observations are symmetrical respective to the actions
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


def test_observations_are_invariant_to_the_player_trained_wt_reset():
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
    max_steps, batch_size, grid_size = 10, 52, 3
    n_steps = max_steps
    envs = init_my_envs(
        max_steps, batch_size, grid_size, same_obs_for_each_player=False
    )

    batch_deltas = [
        i % max_steps if i % 2 == 0 else i % max_steps - 1
        for i in range(batch_size)
    ]

    for env_i, env in enumerate(envs):
        obs = env.reset()
        assert_obs_is_symmetrical(obs, env)
        step_i = 0

        for _ in range(n_steps):
            overwrite_pos(
                step_i,
                batch_deltas,
                max_steps,
                env,
                p_red_pos,
                p_blue_pos,
                c_red_pos,
                c_blue_pos,
            )
            actions = {
                "player_red": [
                    p_red_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
                "player_blue": [
                    p_blue_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
            }
            _, _, _, _ = env.step(actions)

            step_i += 1

            if step_i == max_steps:
                break


def test_observations_are_not_invariant_to_the_player_trained_wt_step():
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
    max_steps, batch_size, grid_size = 10, 52, 3
    n_steps = max_steps
    envs = init_my_envs(
        max_steps, batch_size, grid_size, same_obs_for_each_player=True
    )

    batch_deltas = [
        i % max_steps if i % 2 == 0 else i % max_steps - 1
        for i in range(batch_size)
    ]

    for env_i, env in enumerate(envs):
        _ = env.reset()
        step_i = 0

        for _ in range(n_steps):
            overwrite_pos(
                step_i,
                batch_deltas,
                max_steps,
                env,
                p_red_pos,
                p_blue_pos,
                c_red_pos,
                c_blue_pos,
            )
            actions = {
                "player_red": [
                    p_red_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
                "player_blue": [
                    p_blue_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
            }
            obs, reward, done, info = env.step(actions)

            step_i += 1
            # assert that observations are not
            # symmetrical respective to the
            # actions
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


def test_observations_are_not_invariant_to_the_player_trained_wt_reset():
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
    max_steps, batch_size, grid_size = 10, 52, 3
    n_steps = max_steps
    envs = init_my_envs(
        max_steps, batch_size, grid_size, same_obs_for_each_player=True
    )

    batch_deltas = [
        i % max_steps if i % 2 == 0 else i % max_steps - 1
        for i in range(batch_size)
    ]

    for env_i, env in enumerate(envs):
        obs = env.reset()
        assert_obs_is_not_symmetrical(obs, env)
        step_i = 0

        for _ in range(n_steps):
            overwrite_pos(
                step_i,
                batch_deltas,
                max_steps,
                env,
                p_red_pos,
                p_blue_pos,
                c_red_pos,
                c_blue_pos,
            )
            actions = {
                "player_red": [
                    p_red_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
                "player_blue": [
                    p_blue_act[(step_i + delta) % max_steps]
                    for delta in batch_deltas
                ],
            }
            _, _, _, _ = env.step(actions)

            step_i += 1

            if step_i == max_steps:
                break
