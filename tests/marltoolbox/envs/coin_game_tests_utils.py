import random
import numpy as np


def init_several_envs(classes, **kwargs):
    return [init_env(env_class=class_, **kwargs) for class_ in classes]


def init_env(
    env_class,
    max_steps,
    seed=None,
    grid_size=3,
    players_can_pick_same_coin=True,
    same_obs_for_each_player=False,
    batch_size=None,
):
    config = {
        "max_steps": max_steps,
        "grid_size": grid_size,
        "both_players_can_pick_the_same_coin": players_can_pick_same_coin,
        "same_obs_for_each_player": same_obs_for_each_player,
    }
    if batch_size is not None:
        config["batch_size"] = batch_size
    env = env_class(config)
    env.seed(seed)
    return env


def check_custom_obs(
    obs,
    grid_size,
    batch_size=None,
    n_in_0=1.0,
    n_in_1=1.0,
    n_in_2_and_above=1.0,
    n_layers=4,
):
    assert len(obs) == 2, "two players"
    for player_obs in obs.values():
        if batch_size is None:
            check_single_obs(
                player_obs,
                grid_size,
                n_layers,
                n_in_0,
                n_in_1,
                n_in_2_and_above,
            )
        else:
            for i in range(batch_size):
                check_single_obs(
                    player_obs[i, ...],
                    grid_size,
                    n_layers,
                    n_in_0,
                    n_in_1,
                    n_in_2_and_above,
                )


def check_single_obs(
    player_obs, grid_size, n_layers, n_in_0, n_in_1, n_in_2_and_above
):
    assert player_obs.shape == (grid_size, grid_size, n_layers)
    assert (
        player_obs[..., 0].sum() == n_in_0
    ), f"observe 1 player red in grid: {player_obs[..., 0]}"
    assert (
        player_obs[..., 1].sum() == n_in_1
    ), f"observe 1 player blue in grid: {player_obs[..., 1]}"
    assert (
        player_obs[..., 2:].sum() == n_in_2_and_above
    ), f"observe 1 coin in grid: {player_obs[..., 2:]}"


def assert_logger_buffer_size(env, n_steps):
    assert_attributes_len_equals_value(
        env,
        n_steps,
    )


def assert_attributes_len_equals_value(
    object_,
    value,
    attributes=("red_pick", "red_pick_own", "blue_pick", "blue_pick_own"),
):
    for attribute in attributes:
        assert len(getattr(object_, attribute)) == value


def helper_test_reset(envs, check_obs_fn, **kwargs):
    for env in envs:
        obs = env.reset()
        check_obs_fn(obs, **kwargs)
        assert_logger_buffer_size(env, n_steps=0)


def helper_test_step(envs, check_obs_fn, **kwargs):
    for env in envs:
        obs = env.reset()
        check_obs_fn(obs, **kwargs)
        assert_logger_buffer_size(env, n_steps=0)

        actions = _get_random_action(env, **kwargs)
        obs, reward, done, info = env.step(actions)
        check_obs_fn(obs, **kwargs)
        assert_logger_buffer_size(env, n_steps=1)
        assert not done["__all__"]


def _get_random_action(env, **kwargs):
    if "batch_size" in kwargs.keys():
        actions = _get_random_action_batch(env, kwargs["batch_size"])
    else:
        actions = _get_random_single_action(env)
    return actions


def _get_random_single_action(env):
    actions = {
        policy_id: random.randint(0, env.NUM_ACTIONS - 1)
        for policy_id in env.players_ids
    }
    return actions


def _get_random_action_batch(env, batch_size):
    actions = {
        policy_id: [
            random.randint(0, env.NUM_ACTIONS - 1) for _ in range(batch_size)
        ]
        for policy_id in env.players_ids
    }
    return actions


def helper_test_multiple_steps(envs, n_steps, check_obs_fn, **kwargs):
    for env in envs:
        obs = env.reset()
        check_obs_fn(obs, **kwargs)
        assert_logger_buffer_size(env, n_steps=0)

        for step_i in range(1, n_steps, 1):
            actions = _get_random_action(env, **kwargs)
            obs, reward, done, info = env.step(actions)
            check_obs_fn(obs, **kwargs)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done["__all__"]


def helper_test_multi_ple_episodes(
    envs,
    n_steps,
    max_steps,
    check_obs_fn,
    **kwargs,
):
    for env in envs:
        obs = env.reset()
        check_obs_fn(obs, **kwargs)
        assert_logger_buffer_size(env, n_steps=0)

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = _get_random_action(env, **kwargs)
            obs, reward, done, info = env.step(actions)
            check_obs_fn(obs, **kwargs)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done["__all__"] or (
                step_i == max_steps and done["__all__"]
            )
            if done["__all__"]:
                obs = env.reset()
                check_obs_fn(obs, **kwargs)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0


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
    red_own,
    blue_own,
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
        assert_logger_buffer_size(env, n_steps=0)
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
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done["__all__"] or (
                step_i == max_steps and done["__all__"]
            )

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
                    "pick_own_color", red_own, info, "player_red", delta_err
                )
                assert_not_present_in_dict_or_close_to(
                    "pick_own_color", blue_own, info, "player_blue", delta_err
                )
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
                assert_logger_buffer_size(env, n_steps=0)
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


def assert_not_present_in_dict_or_close_to(
    key, value, info, player, delta_err
):
    if value is None:
        assert key not in info[player]
    else:
        _assert_close_enough(info[player][key], value, delta_err)


def _assert_close_enough(value, target, delta_err):
    assert abs(value - target) < delta_err, (
        f"{abs(value - target)} <" f" {delta_err}"
    )


def _read_actions(
    p_red_act, p_blue_act, step_i, batch_deltas=None, n_steps_in_epi=None
):
    if batch_deltas is not None:
        return _read_actions_batch(
            p_red_act, p_blue_act, step_i, batch_deltas, n_steps_in_epi
        )
    else:
        return _read_single_action(p_red_act, p_blue_act, step_i)


def _read_actions_batch(
    p_red_act, p_blue_act, step_i, batch_deltas, n_steps_in_epi
):
    actions = {
        "player_red": [
            p_red_act[(step_i + delta) % n_steps_in_epi]
            for delta in batch_deltas
        ],
        "player_blue": [
            p_blue_act[(step_i + delta) % n_steps_in_epi]
            for delta in batch_deltas
        ],
    }
    return actions


def _read_single_action(p_red_act, p_blue_act, step_i):
    actions = {
        "player_red": p_red_act[step_i - 1],
        "player_blue": p_blue_act[step_i - 1],
    }
    return actions


def _assert_ssdmmcg_cooperation_items(
    red_coop_fraction,
    blue_coop_fraction,
    red_coop_speed,
    blue_coop_speed,
    info,
    delta_err,
):
    if _is_using_ssdmmcg(
        blue_coop_fraction,
        red_coop_fraction,
        red_coop_speed,
        blue_coop_speed,
    ):

        assert_not_present_in_dict_or_close_to(
            "blue_coop_fraction",
            blue_coop_fraction,
            info,
            "player_blue",
            delta_err,
        )
        assert_not_present_in_dict_or_close_to(
            "red_coop_fraction",
            red_coop_fraction,
            info,
            "player_red",
            delta_err,
        )
        assert_not_present_in_dict_or_close_to(
            "red_coop_speed",
            red_coop_speed,
            info,
            "player_red",
            delta_err,
        )
        assert_not_present_in_dict_or_close_to(
            "blue_coop_speed",
            blue_coop_speed,
            info,
            "player_blue",
            delta_err,
        )


def _is_using_ssdmmcg(
    blue_coop_fraction, red_coop_fraction, red_coop_speed, blue_coop_speed
):
    return (
        blue_coop_fraction is not None
        or red_coop_fraction is not None
        or red_coop_speed is not None
        or blue_coop_speed is not None
    )


def _overwrite_pos_helper(
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
):
    if batch_deltas is not None:
        overwrite_pos_fn(
            step_i,
            batch_deltas,
            max_steps,
            env,
            p_red_pos,
            p_blue_pos,
            c_red_pos,
            c_blue_pos,
            c_red_coin=c_red_coin,
        )
    else:
        overwrite_pos_fn(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            c_red_pos[step_i],
            c_blue_pos[step_i],
            c_red_coin=c_red_coin,
        )


def shift_consistently(list_, step_i, n_steps_in_epi, batch_deltas):
    return [list_[(step_i + delta) % n_steps_in_epi] for delta in batch_deltas]
