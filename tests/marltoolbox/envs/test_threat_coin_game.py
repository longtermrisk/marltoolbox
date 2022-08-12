import numpy as np

from coin_game_tests_utils import (
    init_several_envs,
    _read_actions,
    _assert_close_enough,
    assert_attributes_len_equals_value,
    check_custom_obs,
)
from marltoolbox.envs.coin_game import ThreatCoinGame
from test_coin_game import overwrite_pos


def init_my_envs(
    max_steps,
    grid_size,
    players_can_pick_same_coin=True,
    same_obs_for_each_player=True,
    target_can_neutralize_threat=False,
    add_surrogate_coin=False,
):
    return init_several_envs(
        (ThreatCoinGame,),
        max_steps=max_steps,
        grid_size=grid_size,
        players_can_pick_same_coin=players_can_pick_same_coin,
        same_obs_for_each_player=same_obs_for_each_player,
        target_can_neutralize_threat=target_can_neutralize_threat,
        add_surrogate_coin=add_surrogate_coin,
    )


def overwrite_pos(
    env,
    p_red_pos,
    p_blue_pos,
    coin_pos_regular,
    coin_pos_threat,
    coin_pos_surrogate,
):

    env.red_pos = p_red_pos
    env.blue_pos = p_blue_pos
    env.coin_pos_regular = coin_pos_regular
    env.coin_pos_threat = coin_pos_threat
    env.coin_pos_surrogate = coin_pos_surrogate

    env.red_pos = np.array(env.red_pos)
    env.blue_pos = np.array(env.blue_pos)
    env.coin_pos_regular = np.array(env.coin_pos_regular)
    env.coin_pos_threat = np.array(env.coin_pos_threat)
    env.coin_pos_surrogate = np.array(env.coin_pos_surrogate)


def check_single_obs(players_obs, grid_size, n_layers, n_surrogate_coins=0):
    for one_obs in players_obs.values():
        assert one_obs.shape == (
            grid_size,
            grid_size,
            n_layers,
        ), f"player_obs.shape {one_obs.shape}"
        assert (
            one_obs[..., 0].sum() == 1
        ), f"observe 1 player red in grid: {one_obs[..., 0]}"
        assert (
            one_obs[..., 1].sum() == 1
        ), f"observe 1 player blue in grid: {one_obs[..., 1]}"
        assert (
            one_obs[..., 2].sum() == 1
        ), f"observe 1 player blue in grid: {one_obs[..., 2]}"
        assert (
            one_obs[..., 3].sum() == 1
        ), f"observe 1 player blue in grid: {one_obs[..., 3]}"
        assert (
            one_obs[..., 4].sum() == n_surrogate_coins
        ), f"observe 1 player blue in grid: {one_obs[..., 4]}"


def helper_assert_info(
    n_steps,
    p_red_act,
    p_blue_act,
    envs,
    max_steps,
    p_red_pos,
    p_blue_pos,
    coin_pos_regular,
    coin_pos_threat,
    coin_pos_surrogate,
    threatener_pick_regular,
    target_pick_regular,
    threatener_pick_threat,
    target_pick_threat,
    threatener_pick_surrogate,
    target_pick_surrogate,
    target_pick_regular_by_priority,
    target_pick_threat_by_priority,
    target_pick_surrogate_by_priority,
    threatener_vanilla_reward,
    threatener_surrogate_reward,
    target_vanilla_reward,
    target_surrogate_reward,
    aerr=0.01,
    **check_obs_kwargs,
):
    for env_i, env in enumerate(envs):
        step_i = 0
        obs = env.reset()
        check_single_obs(obs, n_layers=5, **check_obs_kwargs)
        assert_attributes_len_equals_value(
            object_=env,
            value=0,
            attributes=(
                "threatener_pick_regular",
                "target_pick_regular",
                "threatener_pick_threat",
                "target_pick_threat",
                "threatener_pick_surrogate",
                "target_pick_surrogate",
                "target_pick_regular_by_priority",
                "target_pick_threat_by_priority",
                "target_pick_surrogate_by_priority",
            ),
        )
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            coin_pos_regular[step_i],
            coin_pos_threat[step_i],
            coin_pos_surrogate[step_i],
        )

        for _ in range(n_steps):
            actions = _read_actions(
                p_red_act,
                p_blue_act,
                step_i,
                None,
                n_steps_in_epi=max_steps,
            )
            obs, reward, done, info = env.step(actions)
            check_single_obs(obs, n_layers=5, **check_obs_kwargs)
            _assert_close_enough(
                info["player_red"]["vanilla_reward"],
                threatener_vanilla_reward[step_i],
                aerr,
            )
            _assert_close_enough(
                info["player_red"]["surrogate_reward"],
                threatener_surrogate_reward[step_i],
                aerr,
            )
            _assert_close_enough(
                info["player_blue"]["vanilla_reward"],
                target_vanilla_reward[step_i],
                aerr,
            )
            _assert_close_enough(
                info["player_blue"]["surrogate_reward"],
                target_surrogate_reward[step_i],
                aerr,
            )
            step_i += 1
            assert_attributes_len_equals_value(
                object_=env,
                value=step_i,
                attributes=(
                    "threatener_pick_regular",
                    "target_pick_regular",
                    "threatener_pick_threat",
                    "target_pick_threat",
                    "threatener_pick_surrogate",
                    "target_pick_surrogate",
                    "target_pick_regular_by_priority",
                    "target_pick_threat_by_priority",
                    "target_pick_surrogate_by_priority",
                ),
            )
            assert not done["__all__"] or (step_i == max_steps and done["__all__"])

            if done["__all__"]:
                print("info", info)
                print("step_i", step_i)
                print("env", env)
                print("env_i", env_i)
                _assert_close_enough(
                    info["player_red"]["threatener_pick_regular"],
                    threatener_pick_regular,
                    aerr,
                )
                _assert_close_enough(
                    info["player_blue"]["target_pick_regular"],
                    target_pick_regular,
                    aerr,
                )
                _assert_close_enough(
                    info["player_red"]["threatener_pick_threat"],
                    threatener_pick_threat,
                    aerr,
                )
                _assert_close_enough(
                    info["player_blue"]["target_pick_threat"], target_pick_threat, aerr
                )
                _assert_close_enough(
                    info["player_red"]["threatener_pick_surrogate"],
                    threatener_pick_surrogate,
                    aerr,
                )
                _assert_close_enough(
                    info["player_blue"]["target_pick_surrogate"],
                    target_pick_surrogate,
                    aerr,
                )
                _assert_close_enough(
                    info["player_blue"]["target_pick_regular_by_priority"],
                    target_pick_regular_by_priority,
                    aerr,
                )
                _assert_close_enough(
                    info["player_blue"]["target_pick_threat_by_priority"],
                    target_pick_threat_by_priority,
                    aerr,
                )
                _assert_close_enough(
                    info["player_blue"]["target_pick_surrogate_by_priority"],
                    target_pick_surrogate_by_priority,
                    aerr,
                )

                obs = env.reset()
                check_single_obs(obs, n_layers=5, **check_obs_kwargs)
                assert_attributes_len_equals_value(
                    object_=env,
                    value=0,
                    attributes=(
                        "threatener_pick_regular",
                        "target_pick_regular",
                        "threatener_pick_threat",
                        "target_pick_threat",
                        "threatener_pick_surrogate",
                        "target_pick_surrogate",
                        "target_pick_regular_by_priority",
                        "target_pick_threat_by_priority",
                        "target_pick_surrogate_by_priority",
                    ),
                )
                step_i = 0

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                coin_pos_regular[step_i],
                coin_pos_threat[step_i],
                coin_pos_surrogate[step_i],
            )


def test_logged_info__red_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_threat = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [1, 1, 1, 1]
    threatener_surrogate_reward = [1, 1, 1, 1]
    target_vanilla_reward = [0, 0, 0, 0]
    target_surrogate_reward = [0, 0, 0, 0]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=1.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.0,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__blue_pick_red_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_threat = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [0, 0, 0, 0]
    threatener_surrogate_reward = [0, 0, 0, 0]
    target_vanilla_reward = [1, 1, 1, 1]
    target_surrogate_reward = [1, 1, 1, 1]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=1.0,
        threatener_pick_threat=0.0,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__blue_pick_blue_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [0, 0, 0, 0]
    threatener_surrogate_reward = [0, 0, 0, 0]
    target_vanilla_reward = [0, 0, 0, 0]
    target_surrogate_reward = [0, 0, 0, 0]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.0,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__blue_pick_blue_all_the_time_wt_neutralize():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, target_can_neutralize_threat=True)
    threatener_vanilla_reward = [0, 0, 0, 0]
    threatener_surrogate_reward = [0, 0, 0, 0]
    target_vanilla_reward = [0, 0, 0, 0]
    target_surrogate_reward = [0, 0, 0, 0]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.0,
        target_pick_threat=1.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__red_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [-2, -2, -2, -2]
    threatener_surrogate_reward = [-2, -2, -2, -2]
    target_vanilla_reward = [-2, -2, -2, -2]
    target_surrogate_reward = [-2, -2, -2, -2]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=1.0,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [-2, -2, -2, -2]
    threatener_surrogate_reward = [-2, -2, -2, -2]
    target_vanilla_reward = [-2, -2, -2, -2]
    target_surrogate_reward = [-2, -2, -2, -2]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=1.0,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_blue_all_the_time_wt_neutralize():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, target_can_neutralize_threat=True)
    threatener_vanilla_reward = [0, 0, 0, 0]
    threatener_surrogate_reward = [0, 0, 0, 0]
    target_vanilla_reward = [0, 0, 0, 0]
    target_surrogate_reward = [0, 0, 0, 0]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.0,
        target_pick_threat=1.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=1.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_threat = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [0, 0, 0, 0]
    threatener_surrogate_reward = [0, 0, 0, 0]
    target_vanilla_reward = [1, 1, 1, 1]
    target_surrogate_reward = [1, 1, 1, 1]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=1.0,
        threatener_pick_threat=0.0,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=1.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_red_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_threat = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [0, 0, 1, 1]
    threatener_surrogate_reward = [0, 0, 1, 1]
    target_vanilla_reward = [1, 1, 0, 0]
    target_surrogate_reward = [1, 1, 0, 0]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.5,
        target_pick_regular=0.5,
        threatener_pick_threat=0.0,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_blue_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [0, 0, -2, -2]
    threatener_surrogate_reward = [0, 0, -2, -2]
    target_vanilla_reward = [0, 0, -2, -2]
    target_surrogate_reward = [0, 0, -2, -2]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.5,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_blue_half_the_time_wt_neutralize():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, target_can_neutralize_threat=True)
    threatener_vanilla_reward = [0, 0, -2, -2]
    threatener_surrogate_reward = [0, 0, -2, -2]
    target_vanilla_reward = [0, 0, -2, -2]
    target_surrogate_reward = [0, 0, -2, -2]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.5,
        target_pick_threat=0.5,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_blue():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [0, 0, 0, -2]
    threatener_surrogate_reward = [0, 0, 0, -2]
    target_vanilla_reward = [0, 0, 0, -2]
    target_surrogate_reward = [0, 0, 0, -2]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.25,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__both_pick_blue_wt_neutralize():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, target_can_neutralize_threat=True)
    threatener_vanilla_reward = [0, 0, 0, -2]
    threatener_surrogate_reward = [0, 0, 0, -2]
    target_vanilla_reward = [0, 0, 0, -2]
    target_surrogate_reward = [0, 0, 0, -2]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.0,
        target_pick_regular=0.0,
        threatener_pick_threat=0.25,
        target_pick_threat=0.5,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )


def test_logged_info__pick_half_the_time_half_blue_half_red():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_regular = [[2, 2], [1, 1], [2, 2], [1, 1]]
    coin_pos_threat = [[1, 1], [2, 2], [1, 1], [2, 2]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    threatener_vanilla_reward = [0, 0, -2, 1]
    threatener_surrogate_reward = [0, 0, -2, 1]
    target_vanilla_reward = [0, 1, -2, 0]
    target_surrogate_reward = [0, 1, -2, 0]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_regular=coin_pos_regular,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        threatener_pick_regular=0.25,
        target_pick_regular=0.25,
        threatener_pick_threat=0.25,
        target_pick_threat=0.0,
        threatener_pick_surrogate=0.0,
        target_pick_surrogate=0.0,
        target_pick_regular_by_priority=0.0,
        target_pick_threat_by_priority=0.0,
        target_pick_surrogate_by_priority=0.0,
        threatener_vanilla_reward=threatener_vanilla_reward,
        threatener_surrogate_reward=threatener_surrogate_reward,
        target_vanilla_reward=target_vanilla_reward,
        target_surrogate_reward=target_surrogate_reward,
    )
