"""
Coin Game environment.
"""
import gym
import numpy as np

from gym.spaces import Discrete, Tuple
from gym.utils import seeding


# class CoinGameVec:
class CoinGameVec(gym.Env):

    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]

        self.step_count = None
        self.np_random = None

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.step_count = 0
        self.red_coin = self.np_random.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos  = self.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = self.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = self.np_random.randint(self.grid_size, size=2)
            self._generate_coin(i)
        return self._generate_state()

    def _generate_coin(self, i):
        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agents
        success = 0
        while success < 2:
            self.coin_pos[i] = self.np_random.randint(self.grid_size, size=(2))
            success  = 1 - self._same_pos(self.red_pos[i],
                                          self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i],
                                          self.coin_pos[i])

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_state(self):
        state = np.zeros([self.batch_size] + self.ob_space_shape)
        for i in range(self.batch_size):
            state[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1
            state[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1
            if self.red_coin[i]:
                state[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
            else:
                state[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        return state

    def _compute_rewards(self):
        reward_red, reward_blue = [], []
        for i in range(self.batch_size):
            generate = False
            reward_red.append(0)
            reward_blue.append(0)
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -2
                    reward_blue[i] += 1

            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1

            if generate:
                self._generate_coin(i)
        return reward_red, reward_blue

    def step(self, actions):
        for j in range(self.batch_size):
            ac0, ac1 = actions[j]
            assert ac0 in {0, 1, 2, 3} and ac1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = \
                (self.red_pos[j] + self.MOVES[ac0]) % self.grid_size
            self.blue_pos[j] = \
                (self.blue_pos[j] + self.MOVES[ac1]) % self.grid_size

        reward_red, reward_blue = self._compute_rewards()

        reward = [np.array(reward_red), np.array(reward_blue)]
        self.step_count += 1
        done = np.array([
            (self.step_count == self.max_steps) for _ in range(self.batch_size)
        ])
        state = self._generate_state()

        return state, reward, done




class AsymCoinGameVec(CoinGameVec):

    def _compute_rewards(self):
        reward_red, reward_blue = [], []
        for i in range(self.batch_size):
            generate = False
            reward_red.append(0)
            reward_blue.append(0)
            # TODO change the effect of asymmetry into a condition like when red coin picked do +1 for red
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 2
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -1
                    reward_blue[i] += 1

            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1

            if generate:
                self._generate_coin(i)
        return reward_red, reward_blue
