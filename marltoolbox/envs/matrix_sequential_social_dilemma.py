##########
# Part of the code modified from:
# https://github.com/alshedivat/lola/tree/master/lola
##########
import logging
from abc import ABC
from collections import Iterable

# from typing import Dict

import gym.spaces
import numpy as np
from gym.spaces import Discrete, Dict
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from marltoolbox.envs.utils.interfaces import InfoAccumulationInterface
from marltoolbox.envs.utils.mixins import (
    TwoPlayersTwoActionsInfoMixin,
    NPlayersNDiscreteActionsInfoMixin,
)

logger = logging.getLogger(__name__)

PLOT_KEYS = [
    "CC_freq",
    "DD_freq",
    "CD_freq",
    "DC_freq",
]

PLOT_ASSEMBLAGE_TAGS = [
    ("_freq_player_row_mean", "_freq_player_col_mean"),
    ("_freq",),
    ("CC_freq",),
    ("DD_freq",),
    ("CD_freq",),
    ("DC_freq",),
]


class SupportRay1_12_0Mixin:
    def _support_ray_1_12_0(self):
        self._agent_ids = self.players_ids

        self.observation_space = gym.spaces.Dict(
            {k: self.OBSERVATION_SPACE_ for k in self._agent_ids}
        )
        self.action_space = gym.spaces.Dict(
            {k: self.ACTION_SPACE_ for k in self._agent_ids}
        )


class MatrixSequentialSocialDilemma(
    InfoAccumulationInterface, MultiAgentEnv, SupportRay1_12_0Mixin, ABC
):
    """
    A multi-agent abstract class for two player matrix games.

    PAYOFF_MATRIX: Numpy array. Along the dimension N, the action of the
    Nth player change. The last dimension is used to select the player
    whose reward you want to know.

    max_steps: number of step in one episode

    players_ids: list of the RLLib agent id of each player

    output_additional_info: ask the environment to aggregate information
    about the last episode and output them as info at the end of the
    episode.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = None
    NUM_STATES = None
    ACTION_SPACE_ = None
    OBSERVATION_SPACE_ = None
    PAYOFF_MATRIX = None
    NAME = None

    def __init__(self, config: dict = {}):
        super().__init__()

        self._sanity_checks(config)

        self.players_ids = config.get("players_ids", ["player_row", "player_col"])
        self.player_row_id, self.player_col_id = self.players_ids
        self.max_steps = config.get("max_steps", 20)
        self.output_additional_info = config.get("output_additional_info", True)
        self.same_obs_for_each_player = config.get("same_obs_for_each_player", True)

        self.step_count_in_current_episode = None

        # To store info about the fraction of each states
        if self.output_additional_info:
            self._init_info()

        self._support_ray_1_12_0()

    @property
    def ACTION_SPACE(self):
        if not hasattr(self, "action_space"):
            self._support_ray_1_12_0()
        return self.action_space

    @property
    def OBSERVATION_SPACE(self):
        if not hasattr(self, "observation_space"):
            self._support_ray_1_12_0()
        return self.observation_space

    def _sanity_checks(self, config):
        assert self.PAYOFF_MATRIX is not None
        assert self.PAYOFF_MATRIX.shape[0] == self.NUM_ACTIONS
        assert self.PAYOFF_MATRIX.shape[1] == self.NUM_ACTIONS
        assert self.PAYOFF_MATRIX.shape[2] == self.NUM_AGENTS
        assert len(self.PAYOFF_MATRIX.shape) == 3

        if "players_ids" in config:
            assert (
                isinstance(config["players_ids"], Iterable)
                and len(config["players_ids"]) == self.NUM_AGENTS
            )

    def seed(self, seed=None):
        """Seed the PRNG of this space."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count_in_current_episode = 0
        if self.output_additional_info:
            self._reset_info()
        return {
            self.player_row_id: self.NUM_STATES - 1,
            self.player_col_id: self.NUM_STATES - 1,
        }

    def step(self, actions: dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: observations, rewards, done, info
        """
        self.step_count_in_current_episode += 1
        action_player_row = actions[self.player_row_id]
        action_player_col = actions[self.player_col_id]

        if self.output_additional_info:
            self._accumulate_info(action_player_row, action_player_col)

        observations = self._produce_observations(action_player_row, action_player_col)
        rewards = self._get_players_rewards(action_player_row, action_player_col)
        epi_is_done = self.step_count_in_current_episode >= self.max_steps
        if self.step_count_in_current_episode > self.max_steps:
            logger.warning("self.step_count_in_current_episode >= self.max_steps")
        info = self._get_info_for_current_epi(epi_is_done)
        return self._to_RLLib_API(observations, rewards, epi_is_done, info)

    def _produce_observations(self, action_player_row, action_player_col):
        if self.same_obs_for_each_player:
            return self._produce_same_observations_for_each_player(
                action_player_row, action_player_col
            )
        else:
            return self._produce_observations_invariant_to_the_player_trained(
                action_player_row, action_player_col
            )

    def convert_observation_into_actions(self, observation):
        assert self.same_obs_for_each_player

        if observation == self.NUM_STATES - 1:
            return [None] * self.NUM_AGENTS

        return [
            observation // self.NUM_ACTIONS,
            observation % self.NUM_ACTIONS,
        ]

    def _produce_same_observations_for_each_player(
        self, action_player_0: int, action_player_1: int
    ):
        return [
            action_player_0 * self.NUM_ACTIONS + action_player_1,
            action_player_0 * self.NUM_ACTIONS + action_player_1,
        ]

    def _produce_observations_invariant_to_the_player_trained(
        self, action_player_0: int, action_player_1: int
    ):
        """
        We want to be able to use a policy trained as player 1
        for evaluation as player 2 and vice versa.
        """
        return [
            action_player_0 * self.NUM_ACTIONS + action_player_1,
            action_player_1 * self.NUM_ACTIONS + action_player_0,
        ]

    def _get_players_rewards(self, action_player_0: int, action_player_1: int):
        return [
            self.PAYOFF_MATRIX[action_player_0][action_player_1][0],
            self.PAYOFF_MATRIX[action_player_0][action_player_1][1],
        ]

    def _to_RLLib_API(
        self, observations: list, rewards: list, epi_is_done: bool, info: dict
    ):

        observations = {
            self.player_row_id: observations[0],
            self.player_col_id: observations[1],
        }

        rewards = {
            self.player_row_id: rewards[0],
            self.player_col_id: rewards[1],
        }

        if info is None:
            info = {}
        else:
            info = {self.player_row_id: info, self.player_col_id: info}

        done = {
            self.player_row_id: epi_is_done,
            self.player_col_id: epi_is_done,
            "__all__": epi_is_done,
        }

        return observations, rewards, done, info

    def _get_info_for_current_epi(self, epi_is_done):
        if epi_is_done and self.output_additional_info:
            info_for_current_epi = self._get_episode_info()
        else:
            info_for_current_epi = None
        return info_for_current_epi

    def __str__(self):
        return self.NAME

    def enumerate_all_states(self, include_reset_state):
        if include_reset_state and self.max_steps == 1:
            yield self.NUM_STATES - 1
        else:
            for i in range(self.NUM_STATES):
                yield i
                if not include_reset_state and i == self.NUM_STATES - 2:
                    # The reset state is NUM_STATE - 1
                    break


class IteratedMatchingPennies(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Matching Pennies game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[+1, -1], [-1, +1]], [[-1, +1], [+1, -1]]])
    NAME = "IMP"


class IteratedPrisonersDilemma(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Prisoner's Dilemma game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[-1, -1], [-3, +0]], [[+0, -3], [-2, -2]]])
    NAME = "IPD"


class IteratedAsymPrisonersDilemma(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Asymmetric Prisoner's Dilemma game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[+0, -1], [-3, +0]], [[+0, -3], [-2, -2]]])
    NAME = "IPD"


class IteratedStagHunt(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Stag Hunt game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[3, 3], [0, 2]], [[2, 0], [1, 1]]])
    NAME = "IteratedStagHunt"


# class IteratedChicken(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
class IteratedChicken(NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma):

    """
    A two-agent environment for the Chicken game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[+0, +0], [-1.0, +1.0]], [[+1, -1], [-10, -10]]])
    NAME = "IteratedChicken"


class IteratedAsymChicken(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Asymmetric Chicken game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[+2.0, +0], [-1.0, +1.0]], [[+2.5, -1], [-10, -10]]])
    NAME = "AsymmetricIteratedChicken"


class IteratedBoS(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the BoS game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [[[+3.0, +2.0], [+0.0, +0.0]], [[+0.0, +0.0], [+2.0, +3.0]]]
    )
    NAME = "IteratedBoS"


class IteratedAsymBoS(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the BoS game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [[[+4.0, +1.0], [+0.0, +0.0]], [[+0.0, +0.0], [+2.0, +2.0]]]
    )
    NAME = "IteratedAsymBoS"


def define_greed_fear_matrix_game(greed, fear):
    class GreedFearGame(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
        NUM_ACTIONS = 2
        NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
        ACTION_SPACE = Discrete(NUM_ACTIONS)
        OBSERVATION_SPACE_ = Discrete(NUM_STATES)
        R = 3
        P = 1
        T = R + greed
        S = P - fear
        PAYOFF_MATRIX = np.array([[[R, R], [S, T]], [[T, S], [P, P]]])
        NAME = "IteratedGreedFear"

        def __str__(self):
            return f"{self.NAME} with greed={greed} and fear={fear}"

    return GreedFearGame


class IteratedBoSAndPD(
    NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NUM_ACTIONS = 3
    NUM_STATES = NUM_ACTIONS**MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [
            [[3.5, +1], [+0, +0], [-3, +2]],
            [[+0.0, +0], [+1, +3], [-3, +2]],
            [[+2.0, -3], [+2, -3], [-1, -1]],
        ]
    )
    NAME = "IteratedBoSAndPD"


class TwoPlayersCustomizableMatrixGame(
    NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma
):

    NAME = "TwoPlayersCustomizableMatrixGame"

    NUM_ACTIONS = None
    NUM_STATES = None
    ACTION_SPACE = None
    OBSERVATION_SPACE = None
    PAYOFF_MATRIX = None

    def __init__(self, config: dict):
        self.PAYOFF_MATRIX = config["PAYOFF_MATRIX"]
        self.NUM_ACTIONS = config["NUM_ACTIONS"]
        self.ACTION_SPACE = Discrete(self.NUM_ACTIONS)
        self.NUM_STATES = self.NUM_ACTIONS**self.NUM_AGENTS + 1
        self.OBSERVATION_SPACE = Discrete(self.NUM_STATES)

        super().__init__(config)


PLOT_KEYS_2P_3A = [
    "0_0",
    "0_1",
    "0_2",
    "0_3",
    "1_0",
    "1_1",
    "1_2",
    "1_3",
    "2_0",
    "2_1",
    "2_2",
    "2_3",
    "3_0",
    "3_1",
    "3_2",
    "3_3",
]

PLOT_ASSEMBLAGE_TAGS_2P_3A = [
    ("0_0",),
    ("0_1",),
    ("0_2",),
    ("0_3",),
    ("1_0",),
    ("1_1",),
    ("1_2",),
    ("1_3",),
    ("2_0",),
    ("2_1",),
    ("2_2",),
    ("2_3",),
    ("3_0",),
    ("3_1",),
    ("3_2",),
    ("3_3",),
]


class AsymmetricMatrixGame(
    NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NUM_ACTIONS = None
    NUM_ACTIONS_PL0 = 2
    NUM_ACTIONS_PL1 = 3
    ACTION_SPACE_PL0 = Discrete(NUM_ACTIONS_PL0)
    ACTION_SPACE_PL1 = Discrete(NUM_ACTIONS_PL1)
    NUM_STATES = NUM_ACTIONS_PL0 * NUM_ACTIONS_PL1 + 1
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = None
    NAME = "AsymmetricMatrixGame"

    def __init__(self, config):
        super().__init__(config)

        if self.output_additional_info:
            all_possible_joint_actions = []
            for i in range(self.ACTION_SPACE_PL0.n):
                for j in range(self.ACTION_SPACE_PL1.n):
                    all_possible_joint_actions.append([i, j])
            self._init_info(all_possible_joint_actions)

    def _support_ray_1_12_0(self):
        self._agent_ids = self.players_ids
        self.observation_space = gym.spaces.Dict(
            {k: self.OBSERVATION_SPACE for k in self._agent_ids}
        )
        assert len(self._agent_ids) == 2
        self.action_space = gym.spaces.Dict(
            {
                k: self.ACTION_SPACE_PL0 if pl_i == 0 else self.ACTION_SPACE_PL1
                for pl_i, k in enumerate(self._agent_ids)
            }
        )

    def _produce_same_observations_for_each_player(
        self, action_player_0: int, action_player_1: int
    ):
        return [
            action_player_0 * self.NUM_ACTIONS_PL1 + action_player_1,
            action_player_0 * self.NUM_ACTIONS_PL1 + action_player_1,
        ]

    def _produce_observations_invariant_to_the_player_trained(
        self, action_player_0: int, action_player_1: int
    ):
        """
        We want to be able to use a policy trained as player 1
        for evaluation as player 2 and vice versa.
        """
        return [
            action_player_0 * self.NUM_ACTIONS_PL1 + action_player_1,
            action_player_1 * self.NUM_ACTIONS_PL0 + action_player_0,
        ]

    def _sanity_checks(self, config):
        assert self.PAYOFF_MATRIX is not None
        assert self.PAYOFF_MATRIX.shape[0] == self.NUM_ACTIONS_PL0
        assert self.PAYOFF_MATRIX.shape[1] == self.NUM_ACTIONS_PL1
        assert self.PAYOFF_MATRIX.shape[2] == self.NUM_AGENTS
        assert len(self.PAYOFF_MATRIX.shape) == 3

        if "players_ids" in config:
            assert (
                isinstance(config["players_ids"], Iterable)
                and len(config["players_ids"]) == self.NUM_AGENTS
            )

    def _to_RLLib_API(
        self, observations: list, rewards: list, epi_is_done: bool, info: dict
    ):

        observations = {
            self.player_row_id: observations[0],
            self.player_col_id: observations[1],
        }

        rewards = {
            self.player_row_id: rewards[0],
            self.player_col_id: rewards[1],
        }

        if info is None:
            info = {}
        else:
            info = {self.player_row_id: info}

        done = {
            self.player_row_id: epi_is_done,
            self.player_col_id: epi_is_done,
            "__all__": epi_is_done,
        }

        return observations, rewards, done, info

    def convert_observation_into_actions(self, observation):
        assert self.same_obs_for_each_player

        if observation == self.NUM_STATES - 1:
            return [None] * self.NUM_AGENTS

        return [
            observation // self.NUM_ACTIONS_PL1,
            observation % self.NUM_ACTIONS_PL1,
        ]


class ThreatGame(AsymmetricMatrixGame):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NUM_ACTIONS = None
    NUM_ACTIONS_PL0 = 2
    NUM_ACTIONS_PL1 = 3
    ACTION_SPACE_PL0 = Discrete(NUM_ACTIONS_PL0)
    ACTION_SPACE_PL1 = Discrete(NUM_ACTIONS_PL1)
    NUM_STATES = NUM_ACTIONS_PL0 * NUM_ACTIONS_PL1 + 1
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [
            [[-5.0, +5.0], [-5.0, 5.0], [0.0, 0.0]],
            [[-10.0, -2.0], [0, -2.0], [0.0, 0.0]],
        ]
    )
    NAME = "ThreatGame"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        assert self.max_steps == 1


class DemandGame(AsymmetricMatrixGame):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NUM_ACTIONS = None
    NUM_ACTIONS_PL0 = 4
    NUM_ACTIONS_PL1 = 4
    ACTION_SPACE_PL0 = Discrete(NUM_ACTIONS_PL0)
    ACTION_SPACE_PL1 = Discrete(NUM_ACTIONS_PL1)
    NUM_STATES = NUM_ACTIONS_PL0 * NUM_ACTIONS_PL1 + 1
    OBSERVATION_SPACE_ = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [
            [[-3.0, -3.0], [2.0, 0.0], [5.0, -5.0], [5.0, -5.0]],
            [[0.0, 2.0], [1.0, 1.0], [5.0, -5.0], [5.0, -5.0]],
            [[-5.0, 5.0], [-5.0, 5.0], [1.0, 1.0], [2.0, 0.0]],
            [[-5.0, 5.0], [-5.0, 5.0], [0.0, 2.0], [1.0, 1.0]],
        ]
    )
    NAME = "DemandGame"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        assert self.max_steps == 1
