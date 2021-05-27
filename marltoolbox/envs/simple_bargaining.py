##########
# Part of the code modified from:
# https://github.com/julianstastny/openspiel-social-dilemmas/blob/jesse-br/lola_bots_ipd.py
##########

import logging
from abc import ABC
from collections import Iterable
from typing import Dict

import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from marltoolbox.envs.utils.interfaces import InfoAccumulationInterface
from marltoolbox.envs.utils.mixins import NPlayersNContinuousActionsInfoMixin

logger = logging.getLogger(__name__)

PLOT_KEYS = [
    "player0_",
    "player1_",
    "_mean",
    "_std",
]

PLOT_ASSEMBLAGE_TAGS = [
    ("player0_",),
    ("player1_",),
    ("player0_", "player1_"),
    ("_mean",),
    ("_std",),
]


class SimpleBargaining(
    NPlayersNContinuousActionsInfoMixin,
    InfoAccumulationInterface,
    MultiAgentEnv,
    ABC,
):
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    ACTION_SPACE = Box(
        low=0.0,
        high=1.0,
        shape=(NUM_ACTIONS,),
        dtype="float32",
    )
    OBSERVATION_SPACE = Box(
        low=0.0,
        high=1.0,
        shape=(NUM_AGENTS, NUM_ACTIONS),
        dtype="float32",
    )
    NAME = "SimpleBargaining"
    INIT_STATE_VALUE = np.ones(shape=OBSERVATION_SPACE.shape) * 0.0
    G = GAINS_FROM_TRADE_FACTOR = 3.0
    MULTIPLIER = 0.2
    PL0_T0, PL0_T1, PL1_T0, PL1_T1 = np.array([3, 9, 7, 2]) * MULTIPLIER

    def __init__(self, config: Dict = {}):
        # logger.warning("ENV NOT DEBBUGED, NOT TESTED")

        if "players_ids" in config:
            assert (
                isinstance(config["players_ids"], Iterable)
                and len(config["players_ids"]) == self.NUM_AGENTS
            )

        self.players_ids = config.get("players_ids", ["player_0", "player_1"])
        self.player_0_id, self.player_1_id = self.players_ids
        self.max_steps = config.get("max_steps", 1)
        assert self.max_steps == 1
        self.output_additional_info = config.get(
            "output_additional_info", True
        )
        self.step_count_in_current_episode = None

        # To store info about the fraction of each states
        if self.output_additional_info:
            self._init_info()

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count_in_current_episode = 0
        if self.output_additional_info:
            self._reset_info()
        return {
            self.player_0_id: self.INIT_STATE_VALUE,
            self.player_1_id: self.INIT_STATE_VALUE,
        }

    def step(self, actions: dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: observations, rewards, done, info
        """
        self.step_count_in_current_episode += 1
        # print("actions", actions)
        actions_player_0 = actions[self.player_0_id]
        actions_player_1 = actions[self.player_1_id]

        if self.output_additional_info:
            self._accumulate_info(
                pl0_work_on_task0=actions_player_0[0],
                pl0_cutoff=actions_player_0[1],
                pl1_work_on_task0=actions_player_1[0],
                pl1_cutoff=actions_player_1[1],
            )

        observations = self._produce_observations(
            actions_player_0, actions_player_1
        )
        rewards = self._get_players_rewards(actions_player_0, actions_player_1)
        epi_is_done = self.step_count_in_current_episode >= self.max_steps
        if self.step_count_in_current_episode > self.max_steps:
            logger.warning(
                "self.step_count_in_current_episode >= self.max_steps"
            )
        info = self._get_info_for_current_epi(epi_is_done)
        return self._to_RLLib_API(observations, rewards, epi_is_done, info)

    def _produce_observations(self, actions_player_row, actions_player_col):
        one_player_obs = np.array(
            [
                actions_player_row,
                actions_player_col,
            ]
        )
        return [one_player_obs, one_player_obs]

    def _get_players_rewards(
        self, action_player_0: list, action_player_1: list
    ):
        pl0_work_on_task0 = action_player_0[0]
        pl1_work_on_task0 = action_player_1[0]
        pl0_work_on_task1 = 1 - pl0_work_on_task0
        pl1_work_on_task1 = 1 - pl1_work_on_task0
        cutoff_p0 = action_player_0[1]
        cutoff_p1 = action_player_1[1]
        # print("pl0_work_on_task0 - cutoff_p1", pl0_work_on_task0, cutoff_p1)
        # print("pl1_work_on_task1 - cutoff_p0", pl1_work_on_task1, cutoff_p0)
        accept_offer_task0 = (pl0_work_on_task0 - cutoff_p1) > 0
        accept_offer_task1 = (pl1_work_on_task1 - cutoff_p0) > 0
        # accept_offer_task1 = (cutoff_p0 - pl1_work_on_task0) > 0
        accept_offer = accept_offer_task0 and accept_offer_task1
        if not accept_offer:
            return [0.0, 0.0]
        else:

            r_pl0_from_task0 = SimpleBargaining._log_v_plus_one(
                np.power(pl0_work_on_task0 + pl1_work_on_task0, self.PL0_T0)
            )
            r_pl0_from_task1 = SimpleBargaining._log_v_plus_one(
                np.power(
                    # player 1 is more productive at task 1 as seen by player 0
                    pl0_work_on_task1 + self.G * pl1_work_on_task1,
                    self.PL0_T1,
                )
            )
            r_pl1_from_task0 = SimpleBargaining._log_v_plus_one(
                np.power(
                    # player 0 is more productive at task 0 as seen by player 1
                    self.G * pl0_work_on_task0 + pl1_work_on_task0,
                    self.PL1_T0,
                )
            )
            r_pl1_from_task1 = SimpleBargaining._log_v_plus_one(
                np.power(pl0_work_on_task1 + pl1_work_on_task1, self.PL1_T1)
            )
            # print(
            #     "r_pl0_from_task0",
            #     r_pl0_from_task0,
            #     "r_pl0_from_task1",
            #     r_pl0_from_task1,
            # )
            # print(
            #     "r_pl1_from_task0",
            #     r_pl1_from_task0,
            #     "r_pl1_from_task1",
            #     r_pl1_from_task1,
            # )
            reward_player0 = r_pl0_from_task0 + r_pl0_from_task1
            reward_player1 = r_pl1_from_task0 + r_pl1_from_task1
            return [reward_player0, reward_player1]

    @staticmethod
    def _log_v_plus_one(v):
        assert (v + 1) > 0, f"v: {v}"
        return np.log(v + 1)

    def _to_RLLib_API(
        self, observations: list, rewards: list, epi_is_done: bool, info: dict
    ):

        observations = {
            self.player_0_id: observations[0],
            self.player_1_id: observations[1],
        }

        rewards = {
            self.player_0_id: rewards[0],
            self.player_1_id: rewards[1],
        }

        if info is None:
            info = {}
        else:
            info = {self.player_0_id: info, self.player_1_id: info}

        done = {
            self.player_0_id: epi_is_done,
            self.player_1_id: epi_is_done,
            "__all__": epi_is_done,
        }
        # print("observations", observations)
        return observations, rewards, done, info

    def _get_info_for_current_epi(self, epi_is_done):
        if epi_is_done and self.output_additional_info:
            info_for_current_epi = self._get_episode_info()
        else:
            info_for_current_epi = None
        return info_for_current_epi

    def __str__(self):
        return self.NAME


if __name__ == "__main__":
    env = SimpleBargaining({})
    v_range = np.arange(0.01, 0.99, 0.1)
    v_range = np.round(v_range, 2)
    print("v_range", v_range)
    for pl_0_w in v_range:
        for pl_0_c in v_range:
            for pl_1_w in v_range:
                for pl_1_c in v_range:
                    pl_0_w = round(pl_0_w, 2)
                    pl_0_c = round(pl_0_c, 2)
                    pl_1_w = round(pl_1_w, 2)
                    pl_1_c = round(pl_1_c, 2)
                    pl0_a = [pl_0_w, pl_0_c]
                    pl1_a = [pl_1_w, pl_1_c]
                    r_0, r_1 = env._get_players_rewards(pl0_a, pl1_a)
                    r_0 = round(r_0, 2)
                    r_1 = round(r_1, 2)
                    print("act", [pl0_a, pl1_a], "r", [r_0, r_1])

    all_r = np.zeros((len(v_range), len(v_range), 2))
    print("all_r", all_r.shape)
    for i, pl_0_w in enumerate(v_range):
        for j, pl_1_w in enumerate(v_range):
            pl_0_w = round(pl_0_w, 2)
            pl_0_c = 0.0
            pl_1_w = round(pl_1_w, 2)
            pl_1_c = 0.0
            pl0_a = [pl_0_w, pl_0_c]
            pl1_a = [pl_1_w, pl_1_c]
            r_0, r_1 = env._get_players_rewards(pl0_a, pl1_a)
            r_0 = round(r_0, 2)
            r_1 = round(r_1, 2)
            all_r[i, j, :] = [r_0, r_1]

    import matplotlib.pyplot as plt
    from marltoolbox.scripts.plot_meta_policies import (
        heatmap,
        annotate_heatmap,
    )

    # plt.plot(all_r[..., 0])
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    plt.suptitle("SimpleBargaining payoffs")
    _, _ = heatmap(
        all_r[..., 0],
        v_range,
        v_range,
        ax=ax1,
        cmap="YlGn",
        cbarlabel="Reward player 1",
    )
    _, _ = heatmap(
        all_r[..., 1],
        v_range,
        v_range,
        ax=ax2,
        cmap="YlGn",
        cbarlabel="Reward player 2",
    )
    plt.show()
