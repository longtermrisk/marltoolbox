##########
# Code modified from: https://github.com/alshedivat/lola/tree/master/lola
##########

from abc import ABC
from collections import Iterable

import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# Abstract class
class MatrixSocialDilemma(MultiAgentEnv, ABC):
    """
    A two-agent base class environment for matrix games.

    """

    def __init__(self, config: dict):
        """
        PAYOUT_MATRIX: numpy array. Along dim 0 (rows), action of
        player 1 change. Along dim 1 (col), action of player 2 change. (0,0) = (C,C), (1,1) = (D,D)

        reward_randomness: add a normal(mean=0,std=reward_randomness)
        sampled value to each player's rewards

        max_steps: length of an episode
        """

        assert self.PAYOUT_MATRIX is not None
        if "players_ids" in config:
            assert isinstance(config["players_ids"], Iterable) and len(config["players_ids"]) == self.N_AGENTS

        self.players_ids = config.get("players_ids", ["player_row", "player_col"])
        self.player_row_id, self.player_col_id = self.players_ids
        self.max_steps = config.get("max_steps", 20)
        self.reward_randomness = config.get("reward_randomness", 0.0)
        self.get_additional_info = config.get("get_additional_info", True)

        self.step_count = None

        # To store info about the fraction of each states
        if self.get_additional_info:
            self._init_info()

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count = 0
        if self.get_additional_info:
            self._reset_info()
        return {
            self.player_row_id: self.N_STATES - 1,
            self.player_col_id: self.N_STATES - 1
        }

    def step(self, action):
        self.step_count += 1
        ac0, ac1 = action[self.player_row_id], action[self.player_col_id]

        # Store info
        if self.get_additional_info:
            self._accumulate_info(ac0, ac1)

        rewards = {player_id: self.PAYOUT_MATRIX[ac0][ac1][i] for i, player_id in enumerate(self.players_ids)}

        if self.reward_randomness != 0.0:
            rewards = self._add_randomness_to_reward(rewards)

        observations = {
            self.player_row_id: ac0 * self.N_ACTIONS + ac1,
            self.player_col_id: ac1 * self.N_ACTIONS + ac0
        }

        if self.step_count == self.max_steps:
            epi_is_done = True
            if self.get_additional_info:
                info_for_this_epi = self._get_info_summary()
                info = {
                    self.player_row_id: info_for_this_epi,
                    self.player_col_id: info_for_this_epi
                }
            else:
                info = {}
        else:
            epi_is_done = False
            info = {}

        done = {
            self.player_row_id: epi_is_done,
            self.player_col_id: epi_is_done,
            "__all__": epi_is_done,
        }
        return observations, rewards, done, info

    def _add_randomness_to_reward(self, rewards:dict)->dict:
        for key in rewards.keys():
            rewards[key] += float(self.np_random.normal(loc=0, scale=self.reward_randomness))
        return rewards

    def _init_info(self):
        raise NotImplementedError()

    def _reset_info(self):
        raise NotImplementedError()

    def _get_info_summary(self):
        raise NotImplementedError()

    def _accumulate_info(self, ac0, ac1):
        raise NotImplementedError()


class TwoPlayersTwoActionsInfo:

    def _init_info(self):
        self.cc_count = []
        self.dd_count = []
        self.cd_count = []
        self.dc_count = []

    def _reset_info(self):
        self.cc_count.clear()
        self.dd_count.clear()
        self.cd_count.clear()
        self.dc_count.clear()

    def _get_info_summary(self):
        return {
            "CC": sum(self.cc_count) / len(self.cc_count),
            "DD": sum(self.dd_count) / len(self.dd_count),
            "CD": sum(self.cd_count) / len(self.cd_count),
            "DC": sum(self.dc_count) / len(self.dc_count),
        }

    def _accumulate_info(self, ac0, ac1):
        self.cc_count.append(ac0 == 0 and ac1 == 0)
        self.cd_count.append(ac0 == 0 and ac1 == 1)
        self.dc_count.append(ac0 == 1 and ac1 == 0)
        self.dd_count.append(ac0 == 1 and ac1 == 1)


class IteratedMatchingPennies(TwoPlayersTwoActionsInfo, MatrixSocialDilemma):
    """
    A two-agent environment for the Matching Pennies game.
    """
    N_AGENTS = 2
    N_ACTIONS = 2
    N_STATES = N_ACTIONS ** N_AGENTS + 1
    ACTION_SPACE = Discrete(N_ACTIONS)
    OBSERVATION_SPACE = Discrete(N_STATES)
    PAYOUT_MATRIX = np.array([[[+1, -1], [-1, +1]],
                              [[-1, +1], [+1, -1]]])
    NAME = "IMP"


class IteratedPrisonersDilemma(TwoPlayersTwoActionsInfo, MatrixSocialDilemma):
    """
    A two-agent environment for the Prisoner's Dilemma game.
    """

    N_AGENTS = 2
    N_ACTIONS = 2
    N_STATES = N_ACTIONS ** N_AGENTS + 1
    ACTION_SPACE = Discrete(N_ACTIONS)
    OBSERVATION_SPACE = Discrete(N_STATES)
    PAYOUT_MATRIX = np.array([[[-1, -1], [-3, +0]],
                              [[+0, -3], [-2, -2]]])
    NAME = "IPD"

class IteratedStagHunt(TwoPlayersTwoActionsInfo, MatrixSocialDilemma):
    """
    A two-agent environment for the Stag Hunt game.
    """

    N_AGENTS = 2
    N_ACTIONS = 2
    N_STATES = N_ACTIONS ** N_AGENTS + 1
    ACTION_SPACE = Discrete(N_ACTIONS)
    OBSERVATION_SPACE = Discrete(N_STATES)
    PAYOUT_MATRIX = np.array([[[3, 3], [0, 2]],
                              [[2, 0], [1, 1]]])
    NAME = "IteratedStagHunt"


class IteratedChicken(TwoPlayersTwoActionsInfo, MatrixSocialDilemma):
    """
    A two-agent environment for the Chicken game.
    """

    N_AGENTS = 2
    N_ACTIONS = 2
    N_STATES = N_ACTIONS ** N_AGENTS + 1
    ACTION_SPACE = Discrete(N_ACTIONS)
    OBSERVATION_SPACE = Discrete(N_STATES)
    PAYOUT_MATRIX = np.array([[[+0, +0], [-1., +1.]],
                              [[+1, -1], [-10, -10]]])
    NAME = "IteratedChicken"


class BOTS_PD(MatrixSocialDilemma):
    """
    A two-agent environment for the BOTS + PD game.
    """

    N_AGENTS = 2
    N_ACTIONS = 3
    N_STATES = N_ACTIONS ** N_AGENTS + 1
    ACTION_SPACE = Discrete(N_ACTIONS)
    OBSERVATION_SPACE = Discrete(N_STATES)
    PAYOUT_MATRIX = np.array([[[3.5, +1], [+0, +0], [-3, +2]],
                              [[+0., +0], [+1, +3], [-3, +2]],
                              [[+2., -3], [+2, -3], [-1, -1]]])
    NAME = "BOTS+PD"
