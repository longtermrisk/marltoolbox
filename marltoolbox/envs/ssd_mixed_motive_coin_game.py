import logging

import gym
import numpy as np
from ray.rllib.utils import override

from marltoolbox.envs import coin_game

logger = logging.getLogger(__name__)

PLOT_KEYS = coin_game.PLOT_KEYS + [
    "red_coop_speed",
    "blue_coop_speed",
    "red_coop_fraction",
    "blue_coop_fraction"
]

PLOT_ASSEMBLAGE_TAGS = coin_game.PLOT_ASSEMBLAGE_TAGS + [
    ("red_coop_speed", "blue_coop_speed"),
    ("red_coop_fraction", "blue_coop_fraction"),
    ("red_coop_speed_player_red_mean", "blue_coop_speed_player_blue_mean"),
    ("red_coop_fraction_player_red_mean",
     "blue_coop_fraction_player_blue_mean"),
    ("pick_own_color_player_red_mean", "pick_own_color_player_blue_mean",
     "pick_speed_player_red_mean", "pick_speed_player_blue_mean",
     "red_coop_speed_player_red_mean", "blue_coop_speed_player_blue_mean",
     "red_coop_fraction_player_red_mean",
     "blue_coop_fraction_player_blue_mean"),
]


class SSDMixedMotiveCoinGame(coin_game.CoinGame):

    @override(coin_game.CoinGame)
    def __init__(self, config: dict = {}):
        super().__init__(config)

        self.OBSERVATION_SPACE = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, 6),
            dtype="uint8"
        )

    @override(coin_game.CoinGame)
    def _load_config(self, config):
        super()._load_config(config)
        assert self.both_players_can_pick_the_same_coin, \
            "both_players_can_pick_the_same_coin option must be True in " \
            "ssd mixed motive coin game."
        assert self.same_obs_for_each_player, \
            "same_obs_for_each_player option must be True in " \
            "ssd mixed motive coin game."

    @override(coin_game.CoinGame)
    def _randomize_color_and_player_positions(self):
        # Reset coin color and the players and coin positions
        self.red_pos = \
            self.np_random.randint(low=0, high=self.grid_size, size=(2,))
        self.blue_pos = \
            self.np_random.randint(low=0, high=self.grid_size, size=(2,))

        self.red_coin = self.np_random.randint(low=0, high=2)
        self.red_coin_pos = np.zeros(shape=(2,), dtype=np.int8)
        self.blue_coin_pos = np.zeros(shape=(2,), dtype=np.int8)

        self._players_do_not_overlap_at_start()

    @override(coin_game.CoinGame)
    def _generate_coin(self):
        self._switch_between_coin_color_at_each_generation()
        self._wt_coin_pos_different_from_players_and_other_coin()

    def _wt_coin_pos_different_from_players_and_other_coin(self):

        success = 0
        while success < self.NUM_AGENTS + 1:
            self.red_coin_pos = \
                self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(
                self.red_pos, self.red_coin_pos)
            success += 1 - self._same_pos(
                self.blue_pos, self.red_coin_pos)
            success += 1 - self._same_pos(
                self.blue_coin_pos, self.red_coin_pos)

        success = 0
        while success < self.NUM_AGENTS + 1:
            self.blue_coin_pos = \
                self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(
                self.red_pos, self.blue_coin_pos)
            success += 1 - self._same_pos(
                self.blue_pos, self.blue_coin_pos)
            success += 1 - self._same_pos(
                self.blue_coin_pos, self.red_coin_pos)

    @override(coin_game.CoinGame)
    def _generate_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 6))
        obs[self.red_pos[0], self.red_pos[1], 0] = 1
        obs[self.blue_pos[0], self.blue_pos[1], 1] = 1
        if self.red_coin:
            # Feature 4th is for the red cooperative coin
            obs[self.red_coin_pos[0], self.red_coin_pos[1], 4] = 1
            # Feature 3th is for the blue selfish coin
            obs[self.blue_coin_pos[0], self.blue_coin_pos[1], 3] = 1
        else:
            # Feature 2th is for the red selfish coin
            obs[self.red_coin_pos[0], self.red_coin_pos[1], 2] = 1
            # Feature 5th is for the blue cooperative coin
            obs[self.blue_coin_pos[0], self.blue_coin_pos[1], 5] = 1

        obs = self._apply_optional_invariance_to_the_player_trained(obs)
        return obs

    @override(coin_game.CoinGame)
    def _compute_reward(self):

        reward_red = 0.0
        reward_blue = 0.0
        generate_new_coin = False
        red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = \
            False, False, False, False
        picked_red_coop = False
        picked_blue_coop = False

        if self._same_pos(self.red_pos, self.red_coin_pos):
            if self.red_coin and \
                    self._same_pos(self.blue_pos, self.red_coin_pos):
                # Red coin is a coop coin
                generate_new_coin = True
                reward_red += 1.2
                red_pick_any = True
                red_pick_red = True
                blue_pick_any = True
                picked_red_coop = True
            elif not self.red_coin:
                # Red coin is a selfish coin
                generate_new_coin = True
                reward_red += 1.0
                red_pick_any = True
                red_pick_red = True
        elif self._same_pos(self.blue_pos, self.blue_coin_pos):
            if not self.red_coin and \
                    self._same_pos(self.red_pos, self.blue_coin_pos):
                # Blue coin is a coop coin
                generate_new_coin = True
                reward_blue += 1.4
                red_pick_any = True
                blue_pick_any = True
                blue_pick_blue = True
                picked_blue_coop = True
            elif self.red_coin:
                # Blue coin is a selfish coin
                generate_new_coin = True
                reward_blue += 1.0
                blue_pick_any = True
                blue_pick_blue = True

        reward_list = [reward_red, reward_blue]
        if self.output_additional_info:
            self._accumulate_info(red_pick_any=red_pick_any,
                                  red_pick_red=red_pick_red,
                                  blue_pick_any=blue_pick_any,
                                  blue_pick_blue=blue_pick_blue,
                                  picked_red_coop=picked_red_coop,
                                  picked_blue_coop=picked_blue_coop)

        return reward_list, generate_new_coin

    @override(coin_game.CoinGame)
    def _init_info(self):
        self.red_pick = []
        self.red_pick_own = []
        self.blue_pick = []
        self.blue_pick_own = []
        self.picked_red_coop = []
        self.picked_blue_coop = []

    @override(coin_game.CoinGame)
    def _reset_info(self):
        self.red_pick.clear()
        self.red_pick_own.clear()
        self.blue_pick.clear()
        self.blue_pick_own.clear()
        self.picked_red_coop.clear()
        self.picked_blue_coop.clear()

    @override(coin_game.CoinGame)
    def _accumulate_info(
            self, red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue,
            picked_red_coop, picked_blue_coop):

        self.red_pick.append(red_pick_any)
        self.red_pick_own.append(red_pick_red)
        self.blue_pick.append(blue_pick_any)
        self.blue_pick_own.append(blue_pick_blue)
        self.picked_red_coop.append(picked_red_coop)
        self.picked_blue_coop.append(picked_blue_coop)

    @override(coin_game.CoinGame)
    def _get_episode_info(self):
        """
        Output the following information:
        pick_speed is the fraction of steps during which the player picked a
        coin.
        pick_own_color is the fraction of coins picked by the player which have
        the same color as the player.
        """
        player_red_info, player_blue_info = {}, {}
        n_steps_played = len(self.red_pick)
        assert n_steps_played == len(self.blue_pick)
        n_coop = sum(self.picked_blue_coop) + sum(self.picked_red_coop)

        if len(self.red_pick) > 0:
            red_pick = sum(self.red_pick)
            player_red_info["pick_speed"] = red_pick / n_steps_played
            if red_pick > 0:
                player_red_info["pick_own_color"] = \
                    sum(self.red_pick_own) / red_pick

            player_red_info["red_coop_speed"] = \
                sum(self.picked_red_coop) / n_steps_played

            if red_pick > 0:
                player_red_info["red_coop_fraction"] = \
                    n_coop / red_pick

        if len(self.blue_pick) > 0:
            blue_pick = sum(self.blue_pick)
            player_blue_info["pick_speed"] = blue_pick / n_steps_played
            if blue_pick > 0:
                player_blue_info["pick_own_color"] = \
                    sum(self.blue_pick_own) / blue_pick

            player_blue_info["blue_coop_speed"] = \
                sum(self.picked_blue_coop) / n_steps_played

            if blue_pick > 0:
                player_blue_info["blue_coop_fraction"] = \
                    n_coop / blue_pick

        return player_red_info, player_blue_info
