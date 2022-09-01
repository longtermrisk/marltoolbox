import numpy as np

from coin_game_tests_utils import (
    init_several_envs,
    _read_actions,
    _assert_close_enough,
    assert_attributes_len_equals_value,
    check_custom_obs,
)
from marltoolbox.envs.coin_game import ThreatCoinGameV3
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
        (ThreatCoinGameV3,),
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
    coin_pos_give_in,
    coin_pos_not_give_in,
    coin_pos_threat,
    coin_pos_surrogate,
    coin_pos_no_threat,
):

    env.red_pos = p_red_pos
    env.blue_pos = p_blue_pos
    env.coin_pos_give_in = coin_pos_give_in
    env.coin_pos_not_give_in = coin_pos_not_give_in
    env.coin_pos_threat = coin_pos_threat
    env.coin_pos_surrogate = coin_pos_surrogate
    env.coin_pos_no_threat = coin_pos_no_threat

    env.red_pos = np.array(env.red_pos)
    env.blue_pos = np.array(env.blue_pos)
    env.coin_pos_give_in = np.array(env.coin_pos_give_in)
    env.coin_pos_not_give_in = np.array(env.coin_pos_not_give_in)
    env.coin_pos_threat = np.array(env.coin_pos_threat)
    env.coin_pos_surrogate = np.array(env.coin_pos_surrogate)
    env.coin_pos_no_threat = np.array(env.coin_pos_no_threat)


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
            one_obs[..., 4].sum() == 1
        ), f"observe 1 player blue in grid: {one_obs[..., 4]}"
        assert (
            one_obs[..., 5].sum() == 1
        ), f"observe 1 player blue in grid: {one_obs[..., 4]}"
        assert (
            one_obs[..., 6].sum() == 1
        ), f"observe 1 player blue in grid: {one_obs[..., 4]}"


def read_actions(p_red_act, p_blue_act, step_i, batch_deltas=None, n_steps_in_epi=None):
    actions = _read_actions(p_red_act, p_blue_act, step_i, batch_deltas, n_steps_in_epi)
    return {"threatener": actions["player_blue"], "target": actions["player_red"]}


def helper_assert_info(
    n_steps,
    p_red_act,
    p_blue_act,
    envs,
    max_steps,
    p_red_pos,
    p_blue_pos,
    coin_pos_give_in,
    coin_pos_not_give_in,
    coin_pos_threat,
    coin_pos_surrogate,
    coin_pos_no_threat,
    cell_0_0,
    cell_0_1,
    cell_0_2,
    cell_1_0,
    cell_1_1,
    cell_1_2,
    reward_vanilla_target,
    reward_vanilla_threatener,
    surrogate_reward_target,
    surrogate_reward_threatener,
    aerr=0.01,
    **check_obs_kwargs,
):
    for env_i, env in enumerate(envs):
        step_i = 0
        obs = env.reset()
        check_single_obs(obs, n_layers=7, **check_obs_kwargs)
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            coin_pos_give_in[step_i],
            coin_pos_not_give_in[step_i],
            coin_pos_threat[step_i],
            coin_pos_surrogate[step_i],
            coin_pos_no_threat[step_i],
        )

        for _ in range(n_steps):
            actions = read_actions(
                p_red_act,
                p_blue_act,
                step_i,
                None,
                n_steps_in_epi=max_steps,
            )
            obs, reward, done, info = env.step(actions)
            check_single_obs(obs, n_layers=7, **check_obs_kwargs)
            _assert_close_enough(
                info["target"]["reward_vanilla"],
                reward_vanilla_target[step_i],
                aerr,
            )
            _assert_close_enough(
                info["target"]["surrogate_reward"],
                surrogate_reward_target[step_i],
                aerr,
            )
            _assert_close_enough(
                info["threatener"]["reward_vanilla"],
                reward_vanilla_threatener[step_i],
                aerr,
            )
            _assert_close_enough(
                info["threatener"]["surrogate_reward"],
                surrogate_reward_threatener[step_i],
                aerr,
            )

            step_i += 1
            assert not done["__all__"] or (step_i == max_steps and done["__all__"])

            if done["__all__"]:
                print("info", info)
                print("step_i", step_i)
                print("env", env)
                print("env_i", env_i)
                _assert_close_enough(
                    info["target"]["cell_0_0"],
                    cell_0_0[step_i],
                    aerr,
                )
                _assert_close_enough(
                    info["target"]["cell_0_1"],
                    cell_0_1[step_i],
                    aerr,
                )
                _assert_close_enough(
                    info["threatener"]["cell_0_2"],
                    cell_0_2[step_i],
                    aerr,
                )
                _assert_close_enough(
                    info["threatener"]["cell_1_0"],
                    cell_1_0[step_i],
                    aerr,
                )
                _assert_close_enough(
                    info["threatener"]["cell_1_1"],
                    cell_1_1[step_i],
                    aerr,
                )
                _assert_close_enough(
                    info["threatener"]["cell_1_2"],
                    cell_1_2[step_i],
                    aerr,
                )

                obs = env.reset()
                check_single_obs(obs, n_layers=7, **check_obs_kwargs)
                step_i = 0

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                coin_pos_give_in[step_i],
                coin_pos_not_give_in[step_i],
                coin_pos_threat[step_i],
                coin_pos_surrogate[step_i],
                coin_pos_no_threat[step_i],
            )


def test_logged_info__cell_0_0_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    coin_pos_give_in = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_not_give_in = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_threat = [[1, 1], [1, 1], [1, 1], [1, 1]]
    coin_pos_surrogate = [[2, 2], [2, 2], [2, 2], [2, 2]]
    coin_pos_no_threat = [[2, 2], [2, 2], [2, 2], [2, 2]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)
    cell_0_0 = 1.0
    cell_0_1 = 0.0
    cell_0_2 = 0.0
    cell_1_0 = 0.0
    cell_1_1 = 0.0
    cell_1_2 = 0.0
    reward_vanilla_target = [-5, -5, -5, -5]
    reward_vanilla_threatener = [5, 5, 5, 5]
    surrogate_reward_target = [0, 0, 0, 0]
    surrogate_reward_threatener = [0, 0, 0, 0]

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        grid_size=grid_size,
        coin_pos_give_in=coin_pos_give_in,
        coin_pos_not_give_in=coin_pos_not_give_in,
        coin_pos_threat=coin_pos_threat,
        coin_pos_surrogate=coin_pos_surrogate,
        coin_pos_no_threat=coin_pos_no_threat,
        cell_0_0=cell_0_0,
        cell_0_1=cell_0_1,
        cell_0_2=cell_0_2,
        cell_1_0=cell_1_0,
        cell_1_1=cell_1_1,
        cell_1_2=cell_1_2,
        reward_vanilla_target=reward_vanilla_target,
        reward_vanilla_threatener=reward_vanilla_threatener,
        surrogate_reward_target=surrogate_reward_target,
        surrogate_reward_threatener=surrogate_reward_threatener,
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
