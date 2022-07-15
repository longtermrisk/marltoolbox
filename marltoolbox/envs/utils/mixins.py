import logging
from abc import ABC
from typing import List

import numpy as np

from marltoolbox.envs.utils.interfaces import InfoAccumulationInterface

logger = logging.getLogger(__name__)


class TwoPlayersTwoActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in a two player discrete game.
    Logs the frequency of each state.
    """

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

    def _get_episode_info(self):
        return {
            "CC_freq": sum(self.cc_count) / len(self.cc_count),
            "DD_freq": sum(self.dd_count) / len(self.dd_count),
            "CD_freq": sum(self.cd_count) / len(self.cd_count),
            "DC_freq": sum(self.dc_count) / len(self.dc_count),
        }

    def _accumulate_info(self, ac0, ac1):
        self.cc_count.append(ac0 == 0 and ac1 == 0)
        self.cd_count.append(ac0 == 0 and ac1 == 1)
        self.dc_count.append(ac0 == 1 and ac1 == 0)
        self.dd_count.append(ac0 == 1 and ac1 == 1)


class NPlayersNDiscreteActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in N player games with discrete
    actions.
    Logs the frequency of action profiles used
    (action profile: the set of actions used during one step by all players).
    """

    def _init_info(self, all_possible_joint_actions: List[List] = None):
        self.info_counters = {"n_steps_accumulated": 0}
        if all_possible_joint_actions is not None:
            for joint_actions in all_possible_joint_actions:
                self.info_counters[self._get_id_for_actions(joint_actions)] = 0

    def _reset_info(self):
        for k in self.info_counters.keys():
            self.info_counters[k] = 0

    def _get_episode_info(self):
        info = {}
        if self.info_counters["n_steps_accumulated"] > 0:
            for k, v in self.info_counters.items():
                if k != "n_steps_accumulated":
                    info[k] = v / self.info_counters["n_steps_accumulated"]

        return info

    def _accumulate_info(self, *actions):
        id = self._get_id_for_actions(actions)
        if id not in self.info_counters:
            self.info_counters[id] = 0
        self.info_counters[id] += 1
        self.info_counters["n_steps_accumulated"] += 1

    def _get_id_for_actions(self, actions):
        return "_".join([str(a) for a in actions])


class NPlayersNContinuousActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in N player games with continuous
    actions.
    Logs the mean and std of action profiles used
    (action profile: the set of actions used during one step by all players).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning(
            "MIXING NPlayersNContinuousActionsInfoMixin NOT DEBBUGED, NOT TESTED"
        )

    def _init_info(self):
        self.data_accumulated = {}

    def _reset_info(self):
        self.data_accumulated = {}

    def _get_episode_info(self):
        info = {}
        for k, v in self.data_accumulated.items():
            array = np.array(v)
            info[f"{k}_mean"] = array.mean()
            info[f"{k}_std"] = array.std()

        return info

    def _accumulate_info(self, **kwargs_actions):
        for k, v in kwargs_actions.items():
            if k not in self.data_accumulated.keys():
                self.data_accumulated[k] = []
            self.data_accumulated[k].append(v)
