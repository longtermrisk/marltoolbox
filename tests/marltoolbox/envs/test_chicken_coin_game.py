import numpy as np

from coin_game_tests_utils import (
    init_several_envs,
    assert_logger_buffer_size,
    _overwrite_pos_helper,
    _read_actions,
    _assert_close_enough,
    assert_not_present_in_dict_or_close_to,
    _assert_ssdmmcg_cooperation_items,
    assert_attributes_len_equals_value,
)
from marltoolbox.envs.coin_game import ChickenCoinGame
from test_coin_game import overwrite_pos, check_obs


def init_my_envs(
    max_steps,
    grid_size,
    players_can_pick_same_coin=True,
    same_obs_for_each_player=True,
):
    return init_several_envs(
        (ChickenCoinGame,),
        max_steps=max_steps,
        grid_size=grid_size,
        players_can_pick_same_coin=players_can_pick_same_coin,
        same_obs_for_each_player=same_obs_for_each_player,
    )


def helper_assert_info(repetitions=10, **kwargs):
    if "batch_size" in kwargs.keys():
        for _ in range(repetitions):
            batch_deltas = np.random.randint(
                0, kwargs["max_steps"] - 1, size=kwargs["batch_size"]
            )
            helper_assert_info_one_time(batch_deltas=batch_deltas, **kwargs)
    else:
        helper_assert_info_one_time(batch_deltas=None, **kwargs)


def helper_assert_info_one_time(
    n_steps,
    p_red_act,
    p_blue_act,
    envs,
    max_steps,
    p_red_pos,
    p_blue_pos,
    c_red_pos,
    c_blue_pos,
    red_speed,
    blue_speed,
    red_pick_alone,
    blue_pick_alone,
    check_obs_fn,
    overwrite_pos_fn,
    c_red_coin=None,
    batch_deltas=None,
    blue_coop_fraction=None,
    red_coop_fraction=None,
    red_coop_speed=None,
    blue_coop_speed=None,
    delta_err=0.01,
    **check_obs_kwargs,
):
    for env_i, env in enumerate(envs):
        step_i = 0
        obs = env.reset()
        check_obs_fn(obs, **check_obs_kwargs)
        assert_attributes_len_equals_value(
            object_=env,
            value=0,
            attributes=(
                "red_pick",
                "only_red_pick",
                "blue_pick",
                "only_blue_pick",
            ),
        )
        _overwrite_pos_helper(
            batch_deltas,
            overwrite_pos_fn,
            step_i,
            max_steps,
            env,
            p_red_pos,
            p_blue_pos,
            c_red_pos,
            c_blue_pos,
            c_red_coin,
        )

        for _ in range(n_steps):
            actions = _read_actions(
                p_red_act,
                p_blue_act,
                step_i,
                batch_deltas,
                n_steps_in_epi=max_steps,
            )
            step_i += 1
            obs, reward, done, info = env.step(actions)
            check_obs_fn(obs, **check_obs_kwargs)
            assert_attributes_len_equals_value(
                object_=env,
                value=step_i,
                attributes=(
                    "red_pick",
                    "only_red_pick",
                    "blue_pick",
                    "only_blue_pick",
                ),
            )
            assert not done["__all__"] or (step_i == max_steps and done["__all__"])

            if done["__all__"]:
                print("info", info)
                print("step_i", step_i)
                print("env", env)
                print("env_i", env_i)
                _assert_close_enough(
                    info["player_red"]["pick_speed"], red_speed, delta_err
                )
                _assert_close_enough(
                    info["player_blue"]["pick_speed"], blue_speed, delta_err
                )
                assert_not_present_in_dict_or_close_to(
                    "pick_speed_alone",
                    red_pick_alone,
                    info,
                    "player_red",
                    delta_err,
                )
                assert_not_present_in_dict_or_close_to(
                    "pick_speed_alone",
                    blue_pick_alone,
                    info,
                    "player_blue",
                    delta_err,
                )
                # assert_not_present_in_dict_or_close_to(
                #     "pick_own_color", blue_own, info, "player_blue", delta_err
                # )
                _assert_ssdmmcg_cooperation_items(
                    red_coop_fraction,
                    blue_coop_fraction,
                    red_coop_speed,
                    blue_coop_speed,
                    info,
                    delta_err,
                )

                obs = env.reset()
                check_obs_fn(obs, **check_obs_kwargs)
                assert_attributes_len_equals_value(
                    object_=env,
                    value=0,
                    attributes=(
                        "red_pick",
                        "only_red_pick",
                        "blue_pick",
                        "only_blue_pick",
                    ),
                )
                step_i = 0

            _overwrite_pos_helper(
                batch_deltas,
                overwrite_pos_fn,
                step_i,
                max_steps,
                env,
                p_red_pos,
                p_blue_pos,
                c_red_pos,
                c_blue_pos,
                c_red_coin,
            )


def test_logged_info__red_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=1.0,
        blue_speed=0.0,
        red_pick_alone=1.0,
        blue_pick_alone=0.0,
    )


def test_logged_info__blue_pick_red_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=0.0,
        blue_speed=1.0,
        red_pick_alone=0.0,
        blue_pick_alone=1.0,
    )


def test_logged_info__blue_pick_blue_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=0.0,
        blue_speed=1.0,
        red_pick_alone=0.0,
        blue_pick_alone=1.0,
    )


def test_logged_info__red_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=1.0,
        blue_speed=0.0,
        red_pick_alone=1.0,
        blue_pick_alone=0.0,
    )


def test_logged_info__both_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=1.0,
        blue_speed=1.0,
        red_pick_alone=0.0,
        blue_pick_alone=0.0,
    )


def test_logged_info__both_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=1.0,
        blue_speed=1.0,
        red_pick_alone=0.0,
        blue_pick_alone=0.0,
    )


def test_logged_info__both_pick_red_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=0.5,
        blue_speed=0.5,
        red_pick_alone=0.5,
        blue_pick_alone=0.5,
    )


def test_logged_info__both_pick_blue_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=0.5,
        blue_speed=0.5,
        red_pick_alone=0.5,
        blue_pick_alone=0.5,
    )


def test_logged_info__both_pick_blue():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=0.25,
        blue_speed=0.5,
        red_pick_alone=0.25,
        blue_pick_alone=0.50,
    )


def test_logged_info__pick_half_the_time_half_blue_half_red():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], None, [1, 1], None]
    c_blue_pos = [None, [1, 1], None, [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        grid_size=grid_size,
        red_speed=0.5,
        blue_speed=0.5,
        red_pick_alone=0.5,
        blue_pick_alone=0.5,
    )
