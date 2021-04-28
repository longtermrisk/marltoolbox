import copy
import logging
from collections import Iterable

import numpy as np
from numba import jit, prange
from ray.rllib.utils import override

from marltoolbox.envs import vectorized_coin_game
from marltoolbox.envs.vectorized_coin_game import (
    _same_pos,
    move_players,
)
from marltoolbox.envs.vectorized_mixed_motive_coin_game import _place_coin

logger = logging.getLogger(__name__)

PLOT_KEYS = vectorized_coin_game.PLOT_KEYS

PLOT_ASSEMBLAGE_TAGS = vectorized_coin_game.PLOT_ASSEMBLAGE_TAGS


class VectSSDMixedMotiveCG(
    vectorized_coin_game.VectorizedCoinGame,
):
    @override(vectorized_coin_game.VectorizedCoinGame)
    def _load_config(self, config):
        super()._load_config(config)
        assert self.both_players_can_pick_the_same_coin, (
            "both_players_can_pick_the_same_coin option must be True in "
            "ssd mixed motive coin game."
        )
        assert self.same_obs_for_each_player, (
            "same_obs_for_each_player option must be True in "
            "ssd mixed motive coin game."
        )

    @override(vectorized_coin_game.VectorizedCoinGame)
    def _randomize_color_and_player_positions(self):
        # Reset coin color and the players and coin positions
        self.red_pos = np.random.randint(
            self.grid_size, size=(self.batch_size, 2)
        )
        self.blue_pos = np.random.randint(
            self.grid_size, size=(self.batch_size, 2)
        )
        self.red_coin = np.random.randint(
            2, size=self.batch_size, dtype=np.int8
        )
        self.red_coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        self.blue_coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)

        self._players_do_not_overlap_at_start()

    @override(vectorized_coin_game.VectorizedCoinGame)
    def _generate_coin(self):
        self._generate_coin()

    @override(vectorized_coin_game.VectorizedCoinGame)
    def _generate_coin(self):
        generate = np.ones(self.batch_size, dtype=np.int_)
        (
            self.red_coin_pos,
            self.blue_coin_pos,
            self.red_coin,
        ) = generate_coin_wt_numba_optimization(
            self.batch_size,
            generate,
            self.red_coin_pos,
            self.blue_coin_pos,
            self.red_pos,
            self.blue_pos,
            self.grid_size,
            self.red_coin,
        )

    @override(vectorized_coin_game.VectorizedCoinGame)
    def _generate_observation(self):
        obs = generate_observations_wt_numba_optimization(
            self.batch_size,
            self.red_pos,
            self.blue_pos,
            self.red_coin_pos,
            self.blue_coin_pos,
            self.grid_size,
            self.red_coin,
        )

        obs = self._apply_optional_invariance_to_the_player_trained(obs)
        obs, _ = self._optional_unvectorize(obs)
        return obs

    @override(vectorized_coin_game.VectorizedCoinGame)
    def step(self, actions: Iterable):
        actions = self._from_RLLib_API_to_list(actions)
        self.step_count_in_current_episode += 1

        (
            self.red_pos,
            self.blue_pos,
            rewards,
            self.red_coin_pos,
            self.blue_coin_pos,
            observation,
            red_pick_any,
            red_pick_red,
            blue_pick_any,
            blue_pick_blue,
            picked_red_coop,
            picked_blue_coop,
            self.red_coin,
        ) = vectorized_step_wt_numba_optimization(
            actions,
            self.batch_size,
            self.red_pos,
            self.blue_pos,
            self.red_coin_pos,
            self.blue_coin_pos,
            self.grid_size,
            self.red_coin,
        )

        if self.output_additional_info:
            self._accumulate_info(
                red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue
            )

        obs = self._apply_optional_invariance_to_the_player_trained(
            observation
        )
        obs, rewards = self._optional_unvectorize(obs, rewards)

        return self._to_RLLib_API(obs, rewards)

    @override(vectorized_coin_game.VectorizedCoinGame)
    def _save_env(self):
        env_save_state = {
            "red_pos": self.red_pos,
            "blue_pos": self.blue_pos,
            "red_coin_pos": self.red_coin_pos,
            "blue_coin_pos": self.blue_coin_pos,
            "red_coin": self.red_coin,
            "grid_size": self.grid_size,
            "batch_size": self.batch_size,
            "step_count_in_current_episode": self.step_count_in_current_episode,
            "max_steps": self.max_steps,
            "red_pick": self.red_pick,
            "red_pick_own": self.red_pick_own,
            "blue_pick": self.blue_pick,
            "blue_pick_own": self.blue_pick_own,
        }
        return copy.deepcopy(env_save_state)


@jit(nopython=True)
def vectorized_step_wt_numba_optimization(
    actions,
    batch_size,
    red_pos,
    blue_pos,
    red_coin_pos,
    blue_coin_pos,
    grid_size: int,
    red_coin,
):
    red_pos, blue_pos = move_players(
        batch_size, actions, red_pos, blue_pos, grid_size
    )

    (
        reward,
        generate,
        red_pick_any,
        red_pick_red,
        blue_pick_any,
        blue_pick_blue,
        picked_red_coop,
        picked_blue_coop,
    ) = compute_reward(
        batch_size, red_pos, blue_pos, red_coin_pos, blue_coin_pos, red_coin
    )

    (
        red_coin_pos,
        blue_coin_pos,
        red_coin,
    ) = generate_coin_wt_numba_optimization(
        batch_size,
        generate,
        red_coin_pos,
        blue_coin_pos,
        red_pos,
        blue_pos,
        grid_size,
        red_coin,
    )

    obs = generate_observations_wt_numba_optimization(
        batch_size,
        red_pos,
        blue_pos,
        red_coin_pos,
        blue_coin_pos,
        grid_size,
        red_coin,
    )

    return (
        red_pos,
        blue_pos,
        reward,
        red_coin_pos,
        blue_coin_pos,
        obs,
        red_pick_any,
        red_pick_red,
        blue_pick_any,
        blue_pick_blue,
        picked_red_coop,
        picked_blue_coop,
        red_coin,
    )


@jit(nopython=True)
def compute_reward(
    batch_size, red_pos, blue_pos, red_coin_pos, blue_coin_pos, red_coin
):
    reward_red = np.zeros(batch_size)
    reward_blue = np.zeros(batch_size)
    generate = np.zeros(batch_size, dtype=np.int_)
    red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = 0, 0, 0, 0
    picked_red_coop, picked_blue_coop = 0, 0

    for i in prange(batch_size):
        if _same_pos(red_pos[i], red_coin_pos[i]):
            if red_coin[i] and _same_pos(blue_pos[i], red_coin_pos[i]):
                # Red coin is a coop coin
                generate[i] = True
                reward_red[i] += 1.2
                red_pick_any += 1
                red_pick_red += 1
                blue_pick_any += 1
                picked_red_coop += 1
            elif not red_coin[i]:
                # Red coin is a selfish coin
                generate[i] = True
                reward_red[i] += 1.0
                red_pick_any += 1
                red_pick_red += 1
        elif _same_pos(blue_pos[i], blue_coin_pos[i]):
            if not red_coin[i] and _same_pos(red_pos[i], blue_coin_pos[i]):
                # Blue coin is a coop coin
                generate[i] = True
                reward_blue[i] += 2.0
                red_pick_any += 1
                blue_pick_any += 1
                blue_pick_blue += 1
                picked_blue_coop += 1
            elif red_coin[i]:
                # Blue coin is a selfish coin
                generate[i] = True
                reward_blue[i] += 1.0
                blue_pick_any += 1
                blue_pick_blue += 1

    reward = [reward_red, reward_blue]

    return (
        reward,
        generate,
        red_pick_any,
        red_pick_red,
        blue_pick_any,
        blue_pick_blue,
        picked_red_coop,
        picked_blue_coop,
    )


@jit(nopython=True)
def generate_observations_wt_numba_optimization(
    batch_size,
    red_pos,
    blue_pos,
    red_coin_pos,
    blue_coin_pos,
    grid_size,
    red_coin,
):
    obs = np.zeros((batch_size, grid_size, grid_size, 6))
    for i in prange(batch_size):
        obs[i, red_pos[i][0], red_pos[i][1], 0] = 1
        obs[i, blue_pos[i][0], blue_pos[i][1], 1] = 1
        if red_coin[i]:
            # Feature 4th is for the red cooperative coin
            obs[i, red_coin_pos[i][0], red_coin_pos[i][1], 4] = 1
            # Feature 3th is for the blue selfish coin
            obs[i, blue_coin_pos[i][0], blue_coin_pos[i][1], 3] = 1
        else:
            # Feature 2th is for the red selfish coin
            obs[i, red_coin_pos[i][0], red_coin_pos[i][1], 2] = 1
            # Feature 5th is for the blue cooperative coin
            obs[i, blue_coin_pos[i][0], blue_coin_pos[i][1], 5] = 1
    return obs


@jit(nopython=True)
def generate_coin_wt_numba_optimization(
    batch_size,
    generate,
    red_coin_pos,
    blue_coin_pos,
    red_pos,
    blue_pos,
    grid_size,
    red_coin,
):
    for i in prange(batch_size):
        if generate[i]:
            red_coin[i] = 1 - red_coin[i]
            red_coin_pos[i] = _place_coin(
                red_pos[i], blue_pos[i], grid_size, blue_coin_pos[i]
            )
            blue_coin_pos[i] = _place_coin(
                red_pos[i], blue_pos[i], grid_size, red_coin_pos[i]
            )
    return red_coin_pos, blue_coin_pos, red_coin
