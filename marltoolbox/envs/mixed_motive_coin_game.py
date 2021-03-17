import logging

import numpy as np
from ray.rllib.utils import override

from marltoolbox.envs import coin_game

logger = logging.getLogger(__name__)

PLOT_KEYS = coin_game.PLOT_KEYS

PLOT_ASSEMBLAGE_TAGS = coin_game.PLOT_ASSEMBLAGE_TAGS


class MixedMotiveCoinGame(coin_game.CoinGame):

    @override(coin_game.CoinGame)
    def _load_config(self, config):
        super()._load_config(config)
        assert self.both_players_can_pick_the_same_coin, \
            "both_players_can_pick_the_same_coin option must be True in " \
            "mixed motive coin game."

    @override(coin_game.CoinGame)
    def _randomize_color_and_player_positions(self):
        # Reset coin color and the players and coin positions
        self.red_pos = \
            self.np_random.randint(low=0, high=self.grid_size, size=(2,))
        self.blue_pos = \
            self.np_random.randint(low=0, high=self.grid_size, size=(2,))
        self.red_coin_pos = np.zeros(shape=(2,), dtype=np.int8)
        self.blue_coin_pos = np.zeros(shape=(2,), dtype=np.int8)

        self._players_do_not_overlap_at_start()

    @override(coin_game.CoinGame)
    def _generate_coin(self, color_to_generate="both"):
        self._wt_coin_pos_different_from_players_and_other_coin(
            color_to_generate)

    def _wt_coin_pos_different_from_players_and_other_coin(
            self, color_to_generate):

        if color_to_generate == "both" or color_to_generate == "red":
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
        if color_to_generate == "both" or color_to_generate == "blue":
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
        obs = np.zeros((self.grid_size, self.grid_size, 4))
        obs[self.red_pos[0], self.red_pos[1], 0] = 1
        obs[self.blue_pos[0], self.blue_pos[1], 1] = 1
        obs[self.red_coin_pos[0], self.red_coin_pos[1], 2] = 1
        obs[self.blue_coin_pos[0], self.blue_coin_pos[1], 3] = 1

        obs = self._apply_optional_invariance_to_the_player_trained(obs)
        return obs

    @override(coin_game.CoinGame)
    def _compute_reward(self):

        reward_red = 0.0
        reward_blue = 0.0
        generate_new_coin = False
        red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = \
            False, False, False, False

        if self._same_pos(self.red_pos, self.red_coin_pos) and \
                self._same_pos(self.blue_pos, self.red_coin_pos):
            generate_new_coin = "red"
            reward_red += 2
            reward_blue += 1
            red_pick_any = True
            red_pick_red = True
            blue_pick_any = True
        elif self._same_pos(self.red_pos, self.blue_coin_pos) and \
                self._same_pos(self.blue_pos, self.blue_coin_pos):
            generate_new_coin = "blue"
            reward_red += 1
            reward_blue += 2
            red_pick_any = True
            blue_pick_any = True
            blue_pick_blue = True

        reward_list = [reward_red, reward_blue]
        if self.output_additional_info:
            self._accumulate_info(red_pick_any=red_pick_any,
                                  red_pick_red=red_pick_red,
                                  blue_pick_any=blue_pick_any,
                                  blue_pick_blue=blue_pick_blue)

        return reward_list, generate_new_coin
