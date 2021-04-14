"""
Coin Game environment.
"""
from collections import deque

import gym
import numpy as np

# from gym.spaces import self
from gym.utils import seeding


class CoinGameVec(gym.Env):
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
    ]

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]

        self.step_count = None
        self.np_random = None
        self.coin_pick_speed = deque(maxlen=self.max_steps)
        self.rewards_red = deque(maxlen=self.max_steps)
        self.rewards_blue = deque(maxlen=self.max_steps)
        self.player_blue_picked_own = deque(maxlen=self.max_steps)
        self.player_red_picked_own = deque(maxlen=self.max_steps)
        self.player_blue_picked = deque(maxlen=self.max_steps)
        self.player_red_picked = deque(maxlen=self.max_steps)

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count = 0
        self.player_red_picked_own.clear()
        self.player_blue_picked_own.clear()
        self.player_red_picked.clear()
        self.player_blue_picked.clear()
        self.coin_pick_speed.clear()
        self.rewards_red.clear()
        self.rewards_blue.clear()

        self.red_coin = self.np_random.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos = self.np_random.randint(
            self.grid_size, size=(self.batch_size, 2)
        )
        self.blue_pos = self.np_random.randint(
            self.grid_size, size=(self.batch_size, 2)
        )
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = self.np_random.randint(
                    self.grid_size, size=2
                )
            self._generate_coin(i)
        state = self._generate_state()
        # state = np.reshape(state, (self.batch_size, -1))
        observations = [state, state]
        info = [{"available_actions": aa} for aa in self.available_actions]
        info = {"available_actions": info}
        return observations, info

    def _generate_coin(self, i):
        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agent
        success = 0
        while success < 2:
            success = 0
            self.coin_pos[i] = self.np_random.randint(self.grid_size, size=(2))
            success = 1 - self._same_pos(self.red_pos[i], self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i], self.coin_pos[i])

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
        reward_red = np.zeros(self.batch_size)
        reward_blue = np.zeros(self.batch_size)
        generate_count = 0
        coop_blue = 0
        coop_red = 0
        defect_red = 0
        defect_blue = 0
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    coop_red += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -2
                    reward_blue[i] += 1
                    defect_blue += 1
            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                    defect_red += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1
                    coop_blue += 1
            if generate:
                self._generate_coin(i)
                generate_count += 1
        return (
            reward_red,
            reward_blue,
            generate_count,
            coop_blue,
            coop_red,
            defect_red,
            defect_blue,
        )

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        for j in range(self.batch_size):
            a0, a1 = ac0[j], ac1[j]
            assert a0 in {0, 1, 2, 3} and a1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = (
                self.red_pos[j] + self.MOVES[a0]
            ) % self.grid_size
            self.blue_pos[j] = (
                self.blue_pos[j] + self.MOVES[a1]
            ) % self.grid_size

        # Compute rewards
        (
            reward_red,
            reward_blue,
            generate_count,
            coop_blue,
            coop_red,
            defect_red,
            defect_blue,
        ) = self._compute_rewards()

        # Print stuff
        self.player_blue_picked.append(coop_blue + defect_blue)
        self.player_blue_picked_own.append(coop_blue)
        self.player_red_picked.append(coop_red + defect_red)
        self.player_red_picked_own.append(coop_red)
        self.rewards_red.append(sum(reward_red) / len(reward_red))
        self.rewards_blue.append(sum(reward_blue) / len(reward_blue))
        coin_pick_rate = generate_count / self.batch_size
        self.coin_pick_speed.append(coin_pick_rate)

        info = {}
        if self.step_count == self.max_steps:
            coin_pick_speed = sum(self.coin_pick_speed) / self.max_steps
            print("pick_speed", coin_pick_speed)
            info["pick_speed"] = coin_pick_speed

            rewards_per_player = (
                sum(self.rewards_red) / self.max_steps,
                sum(self.rewards_blue) / self.max_steps,
            )
            print("rewards_per_player", rewards_per_player)
            info["rewards_red"] = rewards_per_player[0]
            info["rewards_blue"] = rewards_per_player[1]

            sum_red = sum(self.player_red_picked)
            sum_blue = sum(self.player_blue_picked)
            if sum_red > 0 and sum_blue > 0:
                pick_own_per_player = (
                    sum(self.player_red_picked_own) / sum_red,
                    sum(self.player_blue_picked_own) / sum_blue,
                )
                print("pick_own_per_player", pick_own_per_player)
                info["pick_own_red"] = pick_own_per_player[0]
                info["pick_own_blue"] = pick_own_per_player[1]

            speed_per_player = (
                sum_red / self.max_steps / self.batch_size,
                sum_blue / self.max_steps / self.batch_size,
            )
            print("speed_per_player", speed_per_player)
            info["pick_speed_red"] = speed_per_player[0]
            info["pick_speed_blue"] = speed_per_player[1]

        reward = [reward_red, reward_blue]

        state = self._generate_state()
        observations = [state, state]
        done = self.step_count == self.max_steps
        info.update(
            {
                "available_actions": [
                    {"available_actions": aa} for aa in self.available_actions
                ],
                "generate_rate": coin_pick_rate,
            }
        )
        return observations, reward, done, info


class AsymCoinGameVec(CoinGameVec):
    def _compute_rewards(self):
        reward_red = np.zeros(self.batch_size)
        reward_blue = np.zeros(self.batch_size)
        generate_count = 0
        coop_blue = 0
        coop_red = 0
        defect_red = 0
        defect_blue = 0
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 2
                    coop_red += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -1
                    reward_blue[i] += 1
                    defect_blue += 1
            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                    defect_red += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1
                    coop_blue += 1
            if generate:
                self._generate_coin(i)
                generate_count += 1
        return (
            reward_red,
            reward_blue,
            generate_count,
            coop_blue,
            coop_red,
            defect_red,
            defect_blue,
        )
