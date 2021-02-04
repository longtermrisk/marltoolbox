import random
import copy
import numpy as np

from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame
from envs.utils.wrappers import add_RewardUncertaintyEnvClassWrapper


def init_env(max_steps, env_class, seed=None, grid_size=3):
    config = {
        "max_steps": max_steps,
        "grid_size": grid_size,
    }
    env = env_class(config)
    env.seed(seed)

    return env


def test_add_RewardUncertaintyEnvClassWrapper():
    max_steps, grid_size = 20, 3
    n_steps = int(max_steps * 8.25)
    reward_uncertainty_mean, reward_uncertainty_std = 10, 1
    MyCoinGame = add_RewardUncertaintyEnvClassWrapper(CoinGame, reward_uncertainty_std, reward_uncertainty_mean)
    MyAsymCoinGame = add_RewardUncertaintyEnvClassWrapper(AsymCoinGame, reward_uncertainty_std, reward_uncertainty_mean)
    coin_game = init_env(max_steps, MyCoinGame, grid_size)
    asymm_coin_game = init_env(max_steps, MyAsymCoinGame, grid_size)

    all_rewards = []
    for env in [coin_game, asymm_coin_game]:
        obs = env.reset()

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = {policy_id: random.randint(0, env.NUM_ACTIONS - 1) for policy_id in env.players_ids}
            obs, reward, done, info = env.step(actions)
            print("reward", reward)
            all_rewards.append(reward[env.player_red_id])
            all_rewards.append(reward[env.player_blue_id])

            if done["__all__"]:
                obs = env.reset()
                step_i = 0

    assert np.array(all_rewards).mean() > reward_uncertainty_mean - 1.0
    assert np.array(all_rewards).mean() < reward_uncertainty_mean + 1.0

    assert np.array(all_rewards).std() > reward_uncertainty_std - 0.1
    assert np.array(all_rewards).std() < reward_uncertainty_mean + 0.1


#
# def test_add_SaveLoadEnvClassWrapper():
#     max_steps, grid_size = 20, 3
#     n_steps = int(max_steps * 8.25)
#     MyCoinGame = add_SaveLoadEnvClassWrapper(CoinGame)
#     MyAsymCoinGame = add_SaveLoadEnvClassWrapper(AsymCoinGame)
#     coin_game = init_env(max_steps, MyCoinGame, grid_size)
#     asymm_coin_game = init_env(max_steps, MyAsymCoinGame, grid_size)
#
#     for env in [coin_game, asymm_coin_game]:
#         obs = env.reset()
#         initial_env_state = env._save_env()
#         initial_env_state_saved = copy.deepcopy(initial_env_state)
#         env_initial = copy.deepcopy(env)
#
#         step_i = 0
#         for _ in range(n_steps):
#             step_i += 1
#             actions = {policy_id: random.randint(0, env.NUM_ACTIONS - 1) for policy_id in env.players_ids}
#             obs, reward, done, info = env.step(actions)
#
#             assert all([v == initial_env_state_saved[k]
#                         if not isinstance(v, np.ndarray)
#                         else (v == initial_env_state_saved[k]).all()
#                         for k, v in initial_env_state.items()])
#             env_state_after_step = env._save_env()
#             env_after_step = copy.deepcopy(env)
#
#             env._set_env_state(initial_env_state)
#             env_vars, env_initial_vars = vars(env), vars(env_initial)
#             env_vars.pop("np_random", None)
#             env_initial_vars.pop("np_random", None)
#             assert all([v == env_initial_vars[k]
#                         if not isinstance(v, np.ndarray)
#                         else (v == env_initial_vars[k]).all()
#                         for k, v in env_vars.items()])
#
#             env._set_env_state(env_state_after_step)
#             env_vars, env_after_step_vars = vars(env), vars(env_after_step)
#             env_vars.pop("np_random", None)
#             env_after_step_vars.pop("np_random", None)
#             assert all([v == env_after_step_vars[k]
#                         if not isinstance(v, np.ndarray)
#                         else (v == env_after_step_vars[k]).all()
#                         for k, v in env_vars.items()])
#
#             if done["__all__"]:
#                 obs = env.reset()
#                 step_i = 0
