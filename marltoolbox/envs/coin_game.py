from collections import Iterable
from typing import Dict

import gym
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from numba.typed import List
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class CoinGame(MultiAgentEnv, gym.Env):
    """
    Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NAME = "CoinGame"
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = None
    MOVES = List([
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
    ])

    def __init__(self, config={}):

        if "players_ids" in config:
            assert isinstance(config["players_ids"], Iterable) and len(config["players_ids"]) == self.NUM_AGENTS

        self.players_ids = config.get("players_ids", ["player_red", "player_blue"])
        self.player_red_id, self.player_blue_id = self.players_ids
        self.max_steps = config.get("max_steps", 20)
        self.grid_size = config.get("grid_size", 3)
        self.get_additional_info = config.get("get_additional_info", True)
        self.asymmetric = config.get("asymmetric", False)

        self.OBSERVATION_SPACE = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, 4),
            dtype='uint8'
        )

        self.step_count = None

        if self.get_additional_info:
            self.red_pick = []
            self.red_pick_own = []
            self.blue_pick = []
            self.blue_pick_own = []


    def reset(self):
        self.step_count = 0

        if self.get_additional_info:
            self._reset_info()

        # Reset coin color & players and coin positions
        self.red_coin = np.random.randint(low=0, high=2)
        self.red_pos = np.random.randint(low=0, high=self.grid_size, size=(2,))
        self.blue_pos = np.random.randint(low=0, high=self.grid_size, size=(2,))
        self.coin_pos = np.zeros(shape=(2,), dtype=np.int8)

        # Make sure players don't overlap
        while self._same_pos(self.red_pos, self.blue_pos):
            self.blue_pos = np.random.randint(self.grid_size, size=2)

        self._generate_coin()
        observation = self._generate_observation()

        return {
            self.player_red_id: observation,
            self.player_blue_id: observation
        }


    def step(self, actions: Dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: state, reward, done, info
        """
        actions = self._from_dict_to_list(actions)

        self.step_count += 1
        self._move_players(actions)
        reward_list, generate_new_coin = self._compute_reward()
        if generate_new_coin:
            self._generate_coin()
        observation = self._generate_observation()

        return self._format_to_rllib_API(observation, reward_list)


    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _same_pos(self, x, y):
        return (x == y).all()

    def _move_players(self, actions):
        self.red_pos = (self.red_pos + self.MOVES[actions[0]]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[actions[1]]) % self.grid_size

    def _compute_reward(self):

        reward_red = 0.0
        reward_blue = 0.0
        generate_new_coin = False
        red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = False, False, False, False
        if self.red_coin:
            if self._same_pos(self.red_pos, self.coin_pos):
                generate_new_coin = True
                reward_red += 1
                if self.asymmetric:
                    reward_red += 3
                red_pick_any = True
                red_pick_red = True
            if self._same_pos(self.blue_pos, self.coin_pos):
                generate_new_coin = True
                reward_red += -2
                reward_blue += 1
                blue_pick_any = True
        else:
            if self._same_pos(self.red_pos, self.coin_pos):
                generate_new_coin = True
                reward_red += 1
                reward_blue += -2
                if self.asymmetric:
                    reward_red += 3
                red_pick_any = True
            if self._same_pos(self.blue_pos, self.coin_pos):
                generate_new_coin = True
                reward_blue += 1
                blue_pick_blue = True
                blue_pick_any = True

        reward_list = [reward_red, reward_blue]

        if self.get_additional_info:
            self.red_pick.append(red_pick_any)
            self.red_pick_own.append(red_pick_red)
            self.blue_pick.append(blue_pick_any)
            self.blue_pick_own.append(blue_pick_blue)

        return reward_list, generate_new_coin

    def _generate_coin(self):
        # Switch between red and blue coins
        self.red_coin = 1 - self.red_coin

        # Make sure coin has a different position than the agents
        success = 0
        while success < self.NUM_AGENTS:
            self.coin_pos = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos,self.coin_pos)
            success += 1 - self._same_pos(self.blue_pos,self.coin_pos)


    def _generate_observation(self):
        state = np.zeros((self.grid_size, self.grid_size, 4))
        state[self.red_pos[0], self.red_pos[1], 0] = 1
        state[self.blue_pos[0], self.blue_pos[1], 1] = 1
        if self.red_coin:
            state[self.coin_pos[0], self.coin_pos[1], 2] = 1
        else:
            state[self.coin_pos[0], self.coin_pos[1], 3] = 1
        return state


    def _from_dict_to_list(self, actions):
        """
        Format actions from dict of players to list of lists
        """
        ac_red, ac_blue = actions[self.player_red_id], actions[self.player_blue_id]
        actions = [ac_red, ac_blue]
        return actions

    def _format_to_rllib_API(self, observation, reward):
        state = {
            self.player_red_id: observation,
            self.player_blue_id: observation,
        }
        reward = {
            self.player_red_id: reward[0],
            self.player_blue_id: reward[1],
        }

        epi_is_done = (self.step_count == self.max_steps)
        done = {
            self.player_red_id: epi_is_done,
            self.player_blue_id: epi_is_done,
            "__all__": epi_is_done,
        }

        if epi_is_done and self.get_additional_info:
            info_red, info_blue = self._get_info_summary()
            info = {
                self.player_red_id: info_red,
                self.player_blue_id: info_blue,
            }
        else:
            info = {}

        return state, reward, done, info

    def _get_info_summary(self):
        """
        Output the following information:
        pick_speed is the fraction of steps during which the player picked a coin
        pick_own_color is the fraction of coins picked by the player which have the same color as the player
        """
        red_info, blue_info = {}, {}

        if len(self.red_pick) > 0:
            red_pick = sum(self.red_pick)
            red_info["pick_speed"] = red_pick / len(self.red_pick)
            if red_pick > 0:
                red_info["pick_own_color"] = sum(self.red_pick_own) / red_pick

        if len(self.blue_pick) > 0:
            blue_pick = sum(self.blue_pick)
            blue_info["pick_speed"] = blue_pick / len(self.blue_pick)
            if blue_pick > 0:
                blue_info["pick_own_color"] = sum(self.blue_pick_own) / blue_pick

        return red_info, blue_info

    def _reset_info(self):
        self.red_pick.clear()
        self.red_pick_own.clear()
        self.blue_pick.clear()
        self.blue_pick_own.clear()



class AsymCoinGame(CoinGame):
    NAME = "AsymCoinGame"

    def __init__(self, config={}):
        if "asymmetric" in config:
            assert config["asymmetric"]
        else:
            config["asymmetric"] = True
        super().__init__(config)
