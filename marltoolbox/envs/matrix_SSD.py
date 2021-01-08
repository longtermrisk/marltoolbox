##########
# Part of the code modified from: https://github.com/alshedivat/lola/tree/master/lola
##########

from collections import Iterable

import numpy as np
from abc import ABC
from gym.spaces import Discrete
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# Abstract class
class MatrixSequentialSocialDilemma(MultiAgentEnv, ABC):
    """
    A multi-agent base class environment for matrix games.

    """

    def __init__(self, config: dict):
        """
        PAYOUT_MATRIX: Numpy array. Along dim N the action of the Nth player change.
        reward_randomness: add a Normal(mean=0,std=reward_randomness) variation to each reward
        max_steps: episode length
        players_ids: agent ids for each player
        """

        assert self.PAYOUT_MATRIX is not None
        if "players_ids" in config:
            assert isinstance(config["players_ids"], Iterable) and len(config["players_ids"]) == self.NUM_AGENTS

        self.players_ids = config.get("players_ids", ["player_row", "player_col"])
        self.player_row_id, self.player_col_id = self.players_ids
        self.max_steps = config.get("max_steps", 20)
        self.reward_randomness = config.get("reward_randomness", 0.0)
        self.reward_randomness = 0.0 if self.reward_randomness is None else self.reward_randomness
        self.reward_randomness = 0.0 if self.reward_randomness is False else self.reward_randomness
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

    def _compute_rewards(self, *actions):
        return {player_id: self.PAYOUT_MATRIX[actions[0]][actions[1]][i]
                for i, player_id in enumerate(self.players_ids)}

    def step(self, action):
        self.step_count += 1
        ac0, ac1 = action[self.player_row_id], action[self.player_col_id]

        # Store info
        if self.get_additional_info:
            self._accumulate_info(ac0, ac1)
        rewards = self._compute_rewards(ac0, ac1)

        if self.reward_randomness != 0.0:
            rewards = self._add_randomness_to_reward(rewards)

        observations = {
            self.player_row_id: ac0 * self.NUM_ACTIONS + ac1,
            self.player_col_id: ac1 * self.NUM_ACTIONS + ac0
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

    def _add_randomness_to_reward(self, rewards: dict) -> dict:
        if not hasattr(self, "np_random"):
            self.seed()
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

    def __str__(self):
        return self.NAME


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


class IteratedMatchingPennies(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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


class IteratedPrisonersDilemma(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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


class IteratedAsymPrisonersDilemma(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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


class IteratedStagHunt(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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


class IteratedChicken(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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


class IteratedAsymChicken(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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


class IteratedBoS(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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


class IteratedAsymBoS(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
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
    class GreedFearGame(TwoPlayersTwoActionsInfo, MatrixSequentialSocialDilemma):
        """
        A two-agent environment for the BoS game.
        """

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


class NPlayersDiscreteActionsInfo:

    def _init_info(self):
        self.info_counters = {"n_steps_accumulated": 0}

    def _reset_info(self):
        self.info_counters = {"n_steps_accumulated": 0}

    def _get_info_summary(self):
        info = {}
        if self.info_counters["n_steps_accumulated"] > 0:
            for k, v in self.info_counters.items():
                if k != "n_steps_accumulated":
                    info[k] = v / self.info_counters["n_steps_accumulated"]

        return info

    def _accumulate_info(self, *actions):
        id = "_".join([str(a) for a in actions])
        if id not in self.info_counters:
            self.info_counters[id] = 0
        self.info_counters[id] += 1
        self.info_counters["n_steps_accumulated"] += 1


class IteratedBoSAndPD(NPlayersDiscreteActionsInfo, MatrixSequentialSocialDilemma):
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
