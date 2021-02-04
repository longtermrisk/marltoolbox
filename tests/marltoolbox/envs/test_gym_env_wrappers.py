import copy
import random

import numpy as np

from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame
from marltoolbox.envs.gym_env_wrappers import RewardUncertaintyWrapper


def init_env(max_steps, env_class, reward_uncertainty_mean, reward_uncertainty_std, seed=None, grid_size=3):
    config = {
        "max_steps": max_steps,
        "grid_size": grid_size,
    }
    env = env_class(config)
    env.seed(seed)
    env = RewardUncertaintyWrapper(env,
                             reward_uncertainty_std=reward_uncertainty_std,
                             reward_uncertainty_mean=reward_uncertainty_mean)
    return env


def test_multiple_episodes():
    max_steps, grid_size = 20, 3
    n_steps = int(max_steps * 8.25)
    reward_uncertainty_mean, reward_uncertainty_std = 10, 1
    coin_game = init_env(max_steps, CoinGame, reward_uncertainty_mean, reward_uncertainty_std, grid_size)
    asymm_coin_game = init_env(max_steps, AsymCoinGame, reward_uncertainty_mean, reward_uncertainty_std, grid_size)

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