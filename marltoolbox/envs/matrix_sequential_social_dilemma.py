##########
# Part of the code modified from: https://github.com/alshedivat/lola/tree/master/lola
##########

from collections import Iterable

import numpy as np
from abc import ABC
from gym.spaces import Discrete
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marltoolbox.envs.utils.mixins import TwoPlayersTwoActionsInfoMixin, NPlayersNDiscreteActionsInfoMixin
from marltoolbox.envs.utils.interfaces import InfoAccumulationInterface


class MatrixSequentialSocialDilemma(InfoAccumulationInterface, MultiAgentEnv, ABC):
    """
    A multi-agent abstract class for matrix games.
    """

    def __init__(self, config: dict):
        """
        PAYOUT_MATRIX: Numpy array. Along dim N the action of the Nth player change.
        max_steps: episode length
        players_ids: agent ids for each player
        """

        assert "reward_randomness" not in config.keys()
        assert self.PAYOUT_MATRIX is not None
        if "players_ids" in config:
            assert isinstance(config["players_ids"], Iterable) and len(config["players_ids"]) == self.NUM_AGENTS

        self.players_ids = config.get("players_ids", ["player_row", "player_col"])
        self.player_row_id, self.player_col_id = self.players_ids
        self.max_steps = config.get("max_steps", 20)
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
            self.player_row_id: self.NUM_STATES - 1,
            self.player_col_id: self.NUM_STATES - 1
        }

    def step(self, action):
        self.step_count += 1
        ac0, ac1 = action[self.player_row_id], action[self.player_col_id]

        # Store info
        if self.get_additional_info:
            self._accumulate_info(ac0, ac1)

        return self._to_RLLib_API(ac0, ac1)

    def _to_RLLib_API(self, ac0, ac1):

        observations = {
            self.player_row_id: ac0 * self.NUM_ACTIONS + ac1,
            self.player_col_id: ac1 * self.NUM_ACTIONS + ac0
        }

        rewards = {player_id: self.PAYOUT_MATRIX[ac0][ac1][i]
                        for i, player_id in enumerate(self.players_ids)}

        epi_is_done = self.step_count == self.max_steps

        if epi_is_done and self.get_additional_info:
            info_for_this_epi = self._get_episode_info()
            info = {
                self.player_row_id: info_for_this_epi,
                self.player_col_id: info_for_this_epi
            }
        else:
            info = {}

        done = {
            self.player_row_id: epi_is_done,
            self.player_col_id: epi_is_done,
            "__all__": epi_is_done,
        }

        return observations, rewards, done, info

    def __str__(self):
        return self.NAME

class IteratedMatchingPennies(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Matching Pennies game.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+1, -1], [-1, +1]],
                              [[-1, +1], [+1, -1]]])
    NAME = "IMP"


class IteratedPrisonersDilemma(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Prisoner's Dilemma game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[-1, -1], [-3, +0]],
                              [[+0, -3], [-2, -2]]])
    NAME = "IPD"


class IteratedAsymPrisonersDilemma(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Prisoner's Dilemma game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+0, -1], [-3, +0]],
                              [[+0, -3], [-2, -2]]])
    NAME = "IPD"


class IteratedStagHunt(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Stag Hunt game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[3, 3], [0, 2]],
                              [[2, 0], [1, 1]]])
    NAME = "IteratedStagHunt"


class IteratedChicken(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Chicken game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+0, +0], [-1., +1.]],
                              [[+1, -1], [-10, -10]]])
    NAME = "IteratedChicken"


class IteratedAsymChicken(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Chicken game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+2.0, +0], [-1., +1.]],
                              [[+2.5, -1], [-10, -10]]])
    NAME = "AsymmetricIteratedChicken"


class IteratedBoS(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the BoS game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+3.0, +2.0], [+0.0, +0.0]],
                              [[+0.0, +0.0], [+2.0, +3.0]]])
    NAME = "IteratedBoS"


class IteratedAsymBoS(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the BoS game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+3.5, +1.0], [+0.0, +0.0]],
                              [[+0.0, +0.0], [+1.0, +3.0]]])
    NAME = "IteratedBoS"


def define_greed_fear_matrix_game(greed, fear):
    class GreedFearGame(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):

        NUM_AGENTS = 2
        NUM_ACTIONS = 2
        NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
        ACTION_SPACE = Discrete(NUM_ACTIONS)
        OBSERVATION_SPACE = Discrete(NUM_STATES)
        R = 3
        P = 1
        T = R + greed
        S = P - fear
        PAYOUT_MATRIX = np.array([[[R, R], [S, T]],
                                  [[T, S], [P, P]]])
        NAME = "IteratedGreedFear"

        def __str__(self):
            return f"{self.NAME} with greed={greed} and fear={fear}"

    return GreedFearGame


class IteratedBoSAndPD(NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 3
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[3.5, +1], [+0, +0], [-3, +2]],
                              [[+0., +0], [+1, +3], [-3, +2]],
                              [[+2., -3], [+2, -3], [-1, -1]]])
    NAME = "IteratedBoSAndPD"
