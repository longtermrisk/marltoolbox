import copy
import logging
import random
from collections import Iterable
from typing import Dict

import gym
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override

from marltoolbox.envs.matrix_sequential_social_dilemma import SupportRay1_12_0Mixin
from marltoolbox.envs.utils.interfaces import InfoAccumulationInterface

logger = logging.getLogger(__name__)

PLOT_KEYS = [
    "pick_speed",
    "pick_speed_alone",
    "pick_speed_both",
    "pick_own_color",
    "pick_regular",
    "pick_threat",
    "pick_surrogate",
    "vanilla_reward",
    "surrogate_reward",
    "cell_",
    "target_alone",
    "threatener_alone",
]

PLOT_ASSEMBLAGE_TAGS = [
    ("pick_own_color_player_red_mean", "pick_own_color_player_blue_mean"),
    ("pick_speed_player_red_mean", "pick_speed_player_blue_mean"),
    ("pick_speed_alone_player_red_mean", "pick_speed_alone_player_blue_mean"),
    ("pick_speed_both",),
    ("pick_speed",),
    ("pick_speed_alone",),
    ("pick_speed_both",),
    ("pick_own_color",),
    ("pick_speed", "pick_own_color"),
    (
        "pick_own_color_player_red_mean",
        "pick_own_color_player_blue_mean",
        "pick_speed_player_red_mean",
        "pick_speed_player_blue_mean",
    ),
    ("pick_regular",),
    ("pick_threat",),
    ("pick_surrogate",),
    ("pick_regular", "pick_threat"),
    ("pick_regular", "pick_threat", "pick_surrogate"),
    ("vanilla_reward",),
    ("surrogate_reward",),
    ("vanilla_reward", "surrogate_reward"),
    ("cell_",),
    ("cell_0_",),
    ("cell_1_",),
    ("cell_none",),
    ("target_alone",),
    ("threatener_alone",),
]


class CoinGame(
    InfoAccumulationInterface, MultiAgentEnv, gym.Env, SupportRay1_12_0Mixin
):
    """
    Coin Game environment.
    """

    NAME = "CoinGame"
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE_ = None
    MOVES = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
    ]

    def __init__(self, config: Dict = {}):
        super().__init__()

        self._validate_config(config)

        self._load_config(config)
        self.player_red_id, self.player_blue_id = self.players_ids
        self.n_features = self.grid_size**2 * (2 * self.NUM_AGENTS)
        self.OBSERVATION_SPACE_ = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, 4),
            dtype="uint8",
        )

        self.step_count_in_current_episode = None
        if self.output_additional_info:
            self._init_info()
        self.seed(seed=config.get("seed", None))

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

    def _validate_config(self, config):
        if "players_ids" in config:
            assert isinstance(config["players_ids"], Iterable)
            assert len(config["players_ids"]) == self.NUM_AGENTS

    def _load_config(self, config):
        self.players_ids = config.get("players_ids", ["player_red", "player_blue"])
        self.max_steps = config.get("max_steps", 20)
        self.grid_size = config.get("grid_size", 3)
        self.output_additional_info = config.get("output_additional_info", True)
        self.asymmetric = config.get("asymmetric", False)
        self.both_players_can_pick_the_same_coin = config.get(
            "both_players_can_pick_the_same_coin", True
        )
        self.same_obs_for_each_player = config.get("same_obs_for_each_player", True)
        self.punishment_helped = config.get("punishment_helped", False)

    @override(gym.Env)
    def seed(self, seed=None):
        """Seed the PRNG of this space."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @override(gym.Env)
    def reset(self):
        # print("reset")
        self.step_count_in_current_episode = 0

        if self.output_additional_info:
            self._reset_info()

        self._randomize_color_and_player_positions()
        self._generate_coin()
        obs = self._generate_observation()

        return {self.player_red_id: obs[0], self.player_blue_id: obs[1]}

    def _randomize_color_and_player_positions(self):
        # Reset coin color and the players and coin positions
        self.red_coin = self.np_random.randint(low=0, high=2)
        self.red_pos = self.np_random.randint(low=0, high=self.grid_size, size=(2,))
        self.blue_pos = self.np_random.randint(low=0, high=self.grid_size, size=(2,))

        self._players_do_not_overlap_at_start()

    def _players_do_not_overlap_at_start(self):
        while self._same_pos(self.red_pos, self.blue_pos):
            self.blue_pos = self.np_random.randint(self.grid_size, size=2)

    def _generate_coin(self):
        self._switch_between_coin_color_at_each_generation()
        self._coin_position_different_from_players_positions()

    def _switch_between_coin_color_at_each_generation(self):
        self.red_coin = 1 - self.red_coin

    def _coin_position_different_from_players_positions(self):
        success = 0
        while success < self.NUM_AGENTS:
            self.coin_pos = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos)

    def _generate_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.uint8)
        obs[self.red_pos[0], self.red_pos[1], 0] = 1
        obs[self.blue_pos[0], self.blue_pos[1], 1] = 1
        if self.red_coin:
            obs[self.coin_pos[0], self.coin_pos[1], 2] = 1
        else:
            obs[self.coin_pos[0], self.coin_pos[1], 3] = 1

        obs = self._apply_optional_invariance_to_the_player_trained(obs)
        return obs

    @override(gym.Env)
    def step(self, actions: Dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: obs, rewards, done, info
        """
        actions = self._from_RLLib_API_to_list(actions)

        self.step_count_in_current_episode += 1
        self._move_players(actions)
        reward_list, generate_new_coin = self._compute_reward()
        if generate_new_coin:
            self._generate_coin()
        obs = self._generate_observation()

        return self._to_RLLib_API(obs, reward_list)

    def _same_pos(self, x, y):
        return (x == y).all()

    def _move_players(self, actions):
        self.red_pos = (self.red_pos + self.MOVES[actions[0]]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[actions[1]]) % self.grid_size

    def _compute_reward(self):

        reward_red = 0.0
        reward_blue = 0.0
        generate_new_coin = False
        red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = (
            False,
            False,
            False,
            False,
        )

        red_first_if_both = None
        if not self.both_players_can_pick_the_same_coin:
            if self._same_pos(self.red_pos, self.coin_pos) and self._same_pos(
                self.blue_pos, self.coin_pos
            ):
                red_first_if_both = bool(self.np_random.randint(low=0, high=2))

        if self.red_coin:
            if self._same_pos(self.red_pos, self.coin_pos) and (
                red_first_if_both is None or red_first_if_both
            ):
                generate_new_coin = True
                reward_red += 1
                if self.asymmetric:
                    reward_red += 3
                red_pick_any = True
                red_pick_red = True
            if self._same_pos(self.blue_pos, self.coin_pos) and (
                red_first_if_both is None or not red_first_if_both
            ):
                generate_new_coin = True
                reward_red += -2
                reward_blue += 1
                blue_pick_any = True
                if self.asymmetric and self.punishment_helped:
                    reward_red -= 6
        else:
            if self._same_pos(self.red_pos, self.coin_pos) and (
                red_first_if_both is None or red_first_if_both
            ):
                generate_new_coin = True
                reward_red += 1
                reward_blue += -2
                if self.asymmetric:
                    reward_red += 3
                red_pick_any = True
            if self._same_pos(self.blue_pos, self.coin_pos) and (
                red_first_if_both is None or not red_first_if_both
            ):
                generate_new_coin = True
                reward_blue += 1
                blue_pick_blue = True
                blue_pick_any = True

        reward_list = [reward_red, reward_blue]
        if self.output_additional_info:
            self._accumulate_info(
                red_pick_any=red_pick_any,
                red_pick_red=red_pick_red,
                blue_pick_any=blue_pick_any,
                blue_pick_blue=blue_pick_blue,
            )

        return reward_list, generate_new_coin

    def _from_RLLib_API_to_list(self, actions):
        """
        Format actions from dict of players to list of lists
        """
        actions = [actions[player_id] for player_id in self.players_ids]
        return actions

    def _apply_optional_invariance_to_the_player_trained(self, observation):
        """
        We want to be able to use a policy trained as player 1,
        for evaluation as player 2 and vice versa.
        """

        # player_red_observation contains
        # [Red pos, Blue pos, Red coin pos, Blue coin pos]
        player_red_observation = observation
        # After modification, player_blue_observation will contain
        # [Blue pos, Red pos, Blue coin pos, Red coin pos]
        player_blue_observation = copy.deepcopy(observation)
        if not self.same_obs_for_each_player:
            player_blue_observation[..., 0] = observation[..., 1]
            player_blue_observation[..., 1] = observation[..., 0]
            player_blue_observation[..., 2] = observation[..., 3]
            player_blue_observation[..., 3] = observation[..., 2]
        return [player_red_observation, player_blue_observation]

    def _to_RLLib_API(self, observations, rewards):
        obs = {
            self.player_red_id: observations[0],
            self.player_blue_id: observations[1],
        }
        rewards = {
            self.player_red_id: rewards[0],
            self.player_blue_id: rewards[1],
        }

        epi_is_done = self.step_count_in_current_episode >= self.max_steps
        if self.step_count_in_current_episode > self.max_steps:
            logger.warning(
                "step_count_in_current_episode > self.max_steps: "
                f"{self.step_count_in_current_episode} > {self.max_steps}"
            )

        done = {
            self.player_red_id: epi_is_done,
            self.player_blue_id: epi_is_done,
            "__all__": epi_is_done,
        }

        if epi_is_done and self.output_additional_info:
            player_red_info, player_blue_info = self._get_episode_info()
            info = {
                self.player_red_id: player_red_info,
                self.player_blue_id: player_blue_info,
            }
        else:
            info = {}
        return obs, rewards, done, info

    @override(InfoAccumulationInterface)
    def _get_episode_info(self, n_steps_played=None):
        """
        Output the following information:
        pick_speed is the fraction of steps during which the player picked a
        coin.
        pick_own_color is the fraction of coins picked by the player which have
        the same color as the player.
        """
        player_red_info, player_blue_info = {}, {}
        if n_steps_played is None:
            n_steps_played = len(self.red_pick)
            assert len(self.red_pick) == len(self.blue_pick)

        if len(self.red_pick) > 0:
            red_pick = sum(self.red_pick)
            player_red_info["pick_speed"] = red_pick / n_steps_played
            if red_pick > 0:
                player_red_info["pick_own_color"] = sum(self.red_pick_own) / red_pick

        if len(self.blue_pick) > 0:
            blue_pick = sum(self.blue_pick)
            player_blue_info["pick_speed"] = blue_pick / n_steps_played
            if blue_pick > 0:
                player_blue_info["pick_own_color"] = sum(self.blue_pick_own) / blue_pick

        return player_red_info, player_blue_info

    @override(InfoAccumulationInterface)
    def _reset_info(self):
        self.red_pick.clear()
        self.red_pick_own.clear()
        self.blue_pick.clear()
        self.blue_pick_own.clear()

    @override(InfoAccumulationInterface)
    def _accumulate_info(
        self, red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue
    ):

        self.red_pick.append(red_pick_any)
        self.red_pick_own.append(red_pick_red)
        self.blue_pick.append(blue_pick_any)
        self.blue_pick_own.append(blue_pick_blue)

    @override(InfoAccumulationInterface)
    def _init_info(self):
        self.red_pick = []
        self.red_pick_own = []
        self.blue_pick = []
        self.blue_pick_own = []


class AsymCoinGame(CoinGame):
    NAME = "AsymCoinGame"

    def __init__(self, config: dict = {}):
        if "asymmetric" in config:
            assert config["asymmetric"]
        else:
            config["asymmetric"] = True
        super().__init__(config)


class ChickenCoinGame(CoinGame):
    NAME = "ChickenCoinGame"
    NUM_ACTIONS = 5
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    MOVES = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([0, 0]),
    ]

    def __init__(self, config: dict = {}):
        self._reward_every_n_steps = config.get("reward_every_n_steps", 1)
        if "asymmetric" in config:
            assert config["asymmetric"] == False
        if "both_players_can_pick_the_same_coin" in config:
            assert config["both_players_can_pick_the_same_coin"] == True

        super().__init__(config)

    def _generate_coin(self):
        self._coin_position_different_from_players_positions()

    def _compute_reward(self):

        reward_red = 0.0
        reward_blue = 0.0
        generate_new_coin = False
        red_pick, blue_pick = (False, False)

        if self.step_count_in_current_episode % self._reward_every_n_steps == 0:
            if self._same_pos(self.red_pos, self.coin_pos) and self._same_pos(
                self.blue_pos, self.coin_pos
            ):
                generate_new_coin = True
                reward_red -= 3
                reward_blue -= 3
                red_pick = True
                blue_pick = True
            elif self._same_pos(self.red_pos, self.coin_pos):
                generate_new_coin = True
                reward_red += 1
                reward_blue -= 1
                red_pick = True
            elif self._same_pos(self.blue_pos, self.coin_pos):
                generate_new_coin = True
                reward_red -= 1
                reward_blue += 1
                blue_pick = True

        reward_list = [reward_red, reward_blue]
        if self.output_additional_info:
            self._accumulate_info(
                red_pick=red_pick,
                blue_pick=blue_pick,
                both_pick=red_pick and blue_pick,
                only_red_pick=red_pick and not blue_pick,
                only_blue_pick=not red_pick and blue_pick,
            )

        return reward_list, generate_new_coin

    @override(InfoAccumulationInterface)
    def _get_episode_info(self, n_steps_played=None):
        """
        Output the following information:
        pick_speed is the fraction of steps during which the player picked a
        coin.
        pick_own_color is the fraction of coins picked by the player which have
        the same color as the player.
        """
        player_red_info, player_blue_info = {}, {}
        if n_steps_played is None:
            n_steps_played = len(self.red_pick)
            assert (
                len(self.red_pick)
                == len(self.blue_pick)
                == len(self.only_red_pick)
                == len(self.only_blue_pick)
                == len(self.both_pick)
            )

        if len(self.red_pick) > 0:
            red_pick = sum(self.red_pick)
            player_red_info["pick_speed"] = red_pick / n_steps_played
            only_red_pick = sum(self.only_red_pick)
            player_red_info["pick_speed_alone"] = only_red_pick / n_steps_played

            blue_pick = sum(self.blue_pick)
            player_blue_info["pick_speed"] = blue_pick / n_steps_played
            only_blue_pick = sum(self.only_blue_pick)
            player_blue_info["pick_speed_alone"] = only_blue_pick / n_steps_played

            both_pick = sum(self.both_pick)
            player_red_info["pick_speed_both"] = both_pick / n_steps_played

        return player_red_info, player_blue_info

    @override(InfoAccumulationInterface)
    def _reset_info(self):
        self.red_pick.clear()
        self.blue_pick.clear()
        self.both_pick.clear()
        self.only_red_pick.clear()
        self.only_blue_pick.clear()

    @override(InfoAccumulationInterface)
    def _accumulate_info(
        self, red_pick, blue_pick, both_pick, only_red_pick, only_blue_pick
    ):

        self.red_pick.append(red_pick)
        self.blue_pick.append(blue_pick)
        self.both_pick.append(both_pick)
        self.only_red_pick.append(only_red_pick)
        self.only_blue_pick.append(only_blue_pick)

    @override(InfoAccumulationInterface)
    def _init_info(self):
        self.red_pick = []
        self.blue_pick = []
        self.both_pick = []
        self.only_red_pick = []
        self.only_blue_pick = []


class ThreatCoinGame(CoinGame):
    NAME = "ThreatCoinGame"
    NUM_ACTIONS = 5
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    MOVES = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([0, 0]),
    ]

    def __init__(self, config: dict = {}):
        if "asymmetric" in config:
            assert config["asymmetric"] == False
        if "both_players_can_pick_the_same_coin" in config:
            assert config["both_players_can_pick_the_same_coin"] == True

        super().__init__(config)

        self._target_can_neutralize_threat = config["target_can_neutralize_threat"]
        self._add_surrogate_coin = config["add_surrogate_coin"]
        self._rewards_threats = config.get("rewards_threats", [-2.0, -2.0])
        self._epi_counter = 0
        self.players_ids = config.get("players_ids", ["threatener", "target"])
        self.player_red_id, self.player_blue_id = self.players_ids

        self.OBSERVATION_SPACE_ = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, 5),
            dtype="uint8",
        )
        self._support_ray_1_12_0()

    @override(CoinGame)
    def _generate_coin(self):
        self._wt_coin_pos_different_from_players_and_other_coin()

    @override(gym.Env)
    def step(self, actions: Dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: obs, rewards, done, info
        """
        obs, rewards, done, info = super().step(actions)

        if self.player_red_id in info.keys():
            info[self.player_red_id]["vanilla_reward"] = self.reward_list[0]
            info[self.player_red_id]["surrogate_reward"] = self.surrogate_reward_list[0]
        else:
            info[self.player_red_id] = {
                "vanilla_reward": self.reward_list[0],
                "surrogate_reward": self.surrogate_reward_list[0],
            }

        if self.player_blue_id in info.keys():
            info[self.player_blue_id]["vanilla_reward"] = self.reward_list[1]
            info[self.player_blue_id]["surrogate_reward"] = self.surrogate_reward_list[
                1
            ]
        else:
            info[self.player_blue_id] = {
                "vanilla_reward": self.reward_list[1],
                "surrogate_reward": self.surrogate_reward_list[1],
            }
        return obs, rewards, done, info

    def _wt_coin_pos_different_from_players_and_other_coin(self):

        success = 0
        while success < self.NUM_AGENTS:
            self.coin_pos_regular = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos_regular)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos_regular)

        success = 0
        while success < self.NUM_AGENTS + 1:
            self.coin_pos_threat = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos_threat)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos_threat)
            success += 1 - self._same_pos(self.coin_pos_regular, self.coin_pos_threat)

        if self._add_surrogate_coin:
            success = 0
            while success < self.NUM_AGENTS + 2:
                self.coin_pos_surrogate = self.np_random.randint(self.grid_size, size=2)
                success = 1 - self._same_pos(self.red_pos, self.coin_pos_surrogate)
                success += 1 - self._same_pos(self.blue_pos, self.coin_pos_surrogate)
                success += 1 - self._same_pos(
                    self.coin_pos_regular, self.coin_pos_surrogate
                )
                success += 1 - self._same_pos(
                    self.coin_pos_threat, self.coin_pos_surrogate
                )

    def reset(self):
        self._epi_counter += 1
        return super().reset()

    @override(CoinGame)
    def _generate_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 5), dtype=np.uint8)
        obs[self.red_pos[0], self.red_pos[1], 0] = 1
        obs[self.blue_pos[0], self.blue_pos[1], 1] = 1
        # 2th feature is for the regular coin
        obs[self.coin_pos_regular[0], self.coin_pos_regular[1], 2] = 1
        # 3rd feature is for the threat coin
        obs[self.coin_pos_threat[0], self.coin_pos_threat[1], 3] = 1
        if self._add_surrogate_coin:
            # 4th feature is for the surrogate coin
            obs[self.coin_pos_surrogate[0], self.coin_pos_surrogate[1], 4] = 1

        obs = self._apply_optional_invariance_to_the_player_trained(obs)
        return obs

    def _compute_reward(self):

        reward_threatener, surrogate_reward_threatener = 0.0, 0.0
        reward_target, surrogate_reward_target = 0.0, 0.0
        generate_new_coins = False
        threatener_pick_regular, target_pick_regular = (False, False)
        threatener_pick_threat, target_pick_threat = (False, False)
        threatener_pick_surrogate, target_pick_surrogate = (False, False)
        (
            target_pick_regular_by_priority,
            target_pick_threat_by_priority,
            target_pick_surrogate_by_priority,
        ) = (False, False, False)

        if isinstance(self._rewards_threats[0], list):
            if self._epi_counter < self._rewards_threats[0][0][1]:
                punishment_threatener = self._rewards_threats[0][0][0]
            else:
                punishment_threatener = self._rewards_threats[0][1]
        else:
            punishment_threatener = self._rewards_threats[0]

        if self._same_pos(self.red_pos, self.coin_pos_regular) and self._same_pos(
            self.blue_pos, self.coin_pos_regular
        ):
            generate_new_coins = True
            reward_target += 1
            target_pick_regular = True
            target_pick_regular_by_priority = True
        elif self._same_pos(self.red_pos, self.coin_pos_regular):
            generate_new_coins = True
            reward_threatener += 1
            threatener_pick_regular = True
        elif self._same_pos(self.blue_pos, self.coin_pos_regular):
            generate_new_coins = True
            reward_target += 1
            target_pick_regular = True

        if self._same_pos(self.red_pos, self.coin_pos_threat) and self._same_pos(
            self.blue_pos, self.coin_pos_threat
        ):
            generate_new_coins = True
            if self._target_can_neutralize_threat:
                target_pick_threat_by_priority = True
                target_pick_threat = True
            else:
                reward_threatener += punishment_threatener
                reward_target += self._rewards_threats[1]
                threatener_pick_threat = True
        elif self._same_pos(self.red_pos, self.coin_pos_threat):
            reward_threatener += punishment_threatener
            reward_target += self._rewards_threats[1]
            threatener_pick_threat = True
        elif self._same_pos(self.blue_pos, self.coin_pos_threat):
            if self._target_can_neutralize_threat:
                generate_new_coins = True
                target_pick_threat = True

        surrogate_reward_threatener = reward_threatener
        surrogate_reward_target = reward_target

        if self._add_surrogate_coin:

            if self._same_pos(self.red_pos, self.coin_pos_surrogate) and self._same_pos(
                self.blue_pos, self.coin_pos_surrogate
            ):
                generate_new_coins = True
                if self._target_can_neutralize_threat:
                    target_pick_surrogate_by_priority = True
                    target_pick_surrogate = True
                else:
                    surrogate_reward_threatener += punishment_threatener + 0.1
                    surrogate_reward_target += self._rewards_threats[1]
                    threatener_pick_surrogate = True
            elif self._same_pos(self.red_pos, self.coin_pos_surrogate):
                surrogate_reward_threatener += punishment_threatener + 0.1
                surrogate_reward_target += self._rewards_threats[1]
                threatener_pick_surrogate = True
            elif self._same_pos(self.blue_pos, self.coin_pos_surrogate):
                if self._target_can_neutralize_threat:
                    generate_new_coins = True
                    target_pick_surrogate = True

        reward_list = [reward_threatener, reward_target]
        surrogate_reward_list = [surrogate_reward_threatener, surrogate_reward_target]

        if self.output_additional_info:
            self._accumulate_info(
                threatener_pick_regular=threatener_pick_regular,
                target_pick_regular=target_pick_regular,
                threatener_pick_threat=threatener_pick_threat,
                target_pick_threat=target_pick_threat,
                threatener_pick_surrogate=threatener_pick_surrogate,
                target_pick_surrogate=target_pick_surrogate,
                target_pick_regular_by_priority=target_pick_regular_by_priority,
                target_pick_threat_by_priority=target_pick_threat_by_priority,
                target_pick_surrogate_by_priority=target_pick_surrogate_by_priority,
            )

        self.reward_list = reward_list
        self.surrogate_reward_list = surrogate_reward_list

        if self._add_surrogate_coin:
            return surrogate_reward_list, generate_new_coins
        else:
            return reward_list, generate_new_coins

    @override(InfoAccumulationInterface)
    def _get_episode_info(self, n_steps_played=None):
        """
        Output the following information:
        pick_speed is the fraction of steps during which the player picked a
        coin.
        pick_own_color is the fraction of coins picked by the player which have
        the same color as the player.
        """
        player_red_info, player_blue_info = {}, {}
        if n_steps_played is None:
            n_steps_played = len(self.threatener_pick_regular)
            assert (
                len(self.threatener_pick_regular)
                == len(self.target_pick_regular)
                == len(self.threatener_pick_threat)
                == len(self.target_pick_threat)
                == len(self.threatener_pick_surrogate)
                == len(self.target_pick_surrogate)
                == len(self.target_pick_regular_by_priority)
                == len(self.target_pick_threat_by_priority)
                == len(self.target_pick_surrogate_by_priority)
            )

        if len(self.threatener_pick_regular) > 0:
            player_red_info["threatener_pick_regular"] = (
                sum(self.threatener_pick_regular) / n_steps_played
            )
            player_blue_info["target_pick_regular"] = (
                sum(self.target_pick_regular) / n_steps_played
            )
            player_red_info["threatener_pick_threat"] = (
                sum(self.threatener_pick_threat) / n_steps_played
            )
            player_blue_info["target_pick_threat"] = (
                sum(self.target_pick_threat) / n_steps_played
            )
            player_red_info["threatener_pick_surrogate"] = (
                sum(self.threatener_pick_surrogate) / n_steps_played
            )
            player_blue_info["target_pick_surrogate"] = (
                sum(self.target_pick_surrogate) / n_steps_played
            )
            player_blue_info["target_pick_regular_by_priority"] = (
                sum(self.target_pick_regular_by_priority) / n_steps_played
            )
            player_blue_info["target_pick_threat_by_priority"] = (
                sum(self.target_pick_threat_by_priority) / n_steps_played
            )
            player_blue_info["target_pick_surrogate_by_priority"] = (
                sum(self.target_pick_surrogate_by_priority) / n_steps_played
            )

        return player_red_info, player_blue_info

    @override(InfoAccumulationInterface)
    def _init_info(self):
        self.threatener_pick_regular = []
        self.target_pick_regular = []
        self.threatener_pick_threat = []
        self.target_pick_threat = []
        self.threatener_pick_surrogate = []
        self.target_pick_surrogate = []
        self.target_pick_regular_by_priority = []
        self.target_pick_threat_by_priority = []
        self.target_pick_surrogate_by_priority = []

    @override(InfoAccumulationInterface)
    def _reset_info(self):
        self.threatener_pick_regular.clear()
        self.target_pick_regular.clear()
        self.threatener_pick_threat.clear()
        self.target_pick_threat.clear()
        self.threatener_pick_surrogate.clear()
        self.target_pick_surrogate.clear()
        self.target_pick_regular_by_priority.clear()
        self.target_pick_threat_by_priority.clear()
        self.target_pick_surrogate_by_priority.clear()

    @override(InfoAccumulationInterface)
    def _accumulate_info(
        self,
        threatener_pick_regular,
        target_pick_regular,
        threatener_pick_threat,
        target_pick_threat,
        threatener_pick_surrogate,
        target_pick_surrogate,
        target_pick_regular_by_priority,
        target_pick_threat_by_priority,
        target_pick_surrogate_by_priority,
    ):
        self.threatener_pick_regular.append(threatener_pick_regular)
        self.target_pick_regular.append(target_pick_regular)
        self.threatener_pick_threat.append(threatener_pick_threat)
        self.target_pick_threat.append(target_pick_threat)
        self.threatener_pick_surrogate.append(threatener_pick_surrogate)
        self.target_pick_surrogate.append(target_pick_surrogate)
        self.target_pick_regular_by_priority.append(target_pick_regular_by_priority)
        self.target_pick_threat_by_priority.append(target_pick_threat_by_priority)
        self.target_pick_surrogate_by_priority.append(target_pick_surrogate_by_priority)


class ThreatCoinGameV2(ThreatCoinGame):
    NAME = "ThreatCoinGameV2"

    def __init__(self, config: dict = {}):

        super().__init__(config)
        self._reward_every_n_steps = config.get("reward_every_n_steps", 1)
        self._threat_game_payoff_matrix = config.get(
            "threat_game_payoff_matrix",
            np.array(
                [
                    [[-5.0, +5.0], [-5.0, 5.0], [0.0, 0.0]],
                    [[-10.0, -2.0], [0, -2.0], [0.0, 0.0]],
                ]
            ),
        )
        del self._rewards_threats

    def _compute_reward(self):

        reward_threatener, surrogate_reward_threatener = 0.0, 0.0
        reward_target, surrogate_reward_target = 0.0, 0.0
        generate_new_coins = False
        cell_0_0 = False
        cell_0_1 = False
        cell_0_2 = False
        cell_1_0 = False
        cell_1_1 = False
        cell_1_2 = False

        if self.step_count_in_current_episode % self._reward_every_n_steps == 0:

            if self._same_pos(self.red_pos, self.coin_pos_threat):
                action_threatener = 0
                generate_new_coins = True
            else:
                action_threatener = 2

            if self._same_pos(self.blue_pos, self.coin_pos_regular):
                action_target = 1
                generate_new_coins = True
            else:
                action_target = 0

            reward_target, reward_threatener = self._threat_game_payoff_matrix[
                action_target, action_threatener, :
            ].tolist()

            if self._add_surrogate_coin:
                if self._same_pos(self.red_pos, self.coin_pos_surrogate):
                    assert action_threatener == 2
                    action_threatener = 1
                    generate_new_coins = True

                (
                    surrogate_reward_target,
                    surrogate_reward_threatener,
                ) = self._threat_game_payoff_matrix[
                    action_target, action_threatener, :
                ].tolist()

            if action_target == 0:
                if action_threatener == 0:
                    cell_0_0 = True
                elif action_threatener == 1:
                    cell_0_1 = True
                elif action_threatener == 2:
                    cell_0_2 = True
                else:
                    raise NotImplementedError()
            elif action_target == 1:
                if action_threatener == 0:
                    cell_1_0 = True
                elif action_threatener == 1:
                    cell_1_1 = True
                elif action_threatener == 2:
                    cell_1_2 = True
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            if self.output_additional_info:
                self._accumulate_info(
                    cell_0_0=cell_0_0,
                    cell_0_1=cell_0_1,
                    cell_0_2=cell_0_2,
                    cell_1_0=cell_1_0,
                    cell_1_1=cell_1_1,
                    cell_1_2=cell_1_2,
                )

        reward_list = [reward_threatener, reward_target]
        surrogate_reward_list = [surrogate_reward_threatener, surrogate_reward_target]

        self.reward_list = reward_list
        self.surrogate_reward_list = surrogate_reward_list

        if self._add_surrogate_coin:
            return surrogate_reward_list, generate_new_coins
        else:
            return reward_list, generate_new_coins

    @override(InfoAccumulationInterface)
    def _get_episode_info(self, n_steps_played=None):
        """
        Output the following information:
        pick_speed is the fraction of steps during which the player picked a
        coin.
        pick_own_color is the fraction of coins picked by the player which have
        the same color as the player.
        """
        player_red_info, player_blue_info = {}, {}
        if n_steps_played is None:
            n_steps_played = len(self.cell_0_0)
            assert (
                len(self.cell_0_0)
                == len(self.cell_0_1)
                == len(self.cell_0_2)
                == len(self.cell_1_0)
                == len(self.cell_1_1)
                == len(self.cell_1_2)
            )

        if len(self.cell_0_0) > 0:
            player_blue_info["cell_0_0"] = sum(self.cell_0_0) / n_steps_played
            player_red_info["cell_0_1"] = sum(self.cell_0_1) / n_steps_played
            player_red_info["cell_0_2"] = sum(self.cell_0_2) / n_steps_played
            player_red_info["cell_1_0"] = sum(self.cell_1_0) / n_steps_played
            player_red_info["cell_1_1"] = sum(self.cell_1_1) / n_steps_played
            player_red_info["cell_1_2"] = sum(self.cell_1_2) / n_steps_played
            player_red_info["cell_none"] = sum(self.cell_none) / n_steps_played
            player_red_info["threatener_alone"] = (
                sum(self.threatener_alone) / n_steps_played
            )
            player_red_info["target_alone"] = sum(self.target_alone) / n_steps_played
        return player_red_info, player_blue_info

    @override(InfoAccumulationInterface)
    def _init_info(self):
        self.threatener_alone = []
        self.target_alone = []
        self.cell_0_0 = []
        self.cell_0_1 = []
        self.cell_0_2 = []
        self.cell_1_0 = []
        self.cell_1_1 = []
        self.cell_1_2 = []
        self.cell_none = []

    @override(InfoAccumulationInterface)
    def _reset_info(self):
        self.threatener_alone.clear()
        self.target_alone.clear()
        self.cell_0_0.clear()
        self.cell_0_1.clear()
        self.cell_0_2.clear()
        self.cell_1_0.clear()
        self.cell_1_1.clear()
        self.cell_1_2.clear()
        self.cell_none.clear()

    @override(InfoAccumulationInterface)
    def _accumulate_info(
        self,
        threatener_alone=False,
        target_alone=False,
        cell_0_0=False,
        cell_0_1=False,
        cell_0_2=False,
        cell_1_0=False,
        cell_1_1=False,
        cell_1_2=False,
        cell_none=False,
    ):
        self.threatener_alone.append(threatener_alone)
        self.target_alone.append(target_alone)
        self.cell_0_0.append(cell_0_0)
        self.cell_0_1.append(cell_0_1)
        self.cell_0_2.append(cell_0_2)
        self.cell_1_0.append(cell_1_0)
        self.cell_1_1.append(cell_1_1)
        self.cell_1_2.append(cell_1_2)
        self.cell_none.append(cell_none)


class ThreatCoinGameV3(ThreatCoinGameV2):
    NAME = "ThreatCoinGameV3"

    def __init__(self, config: dict = {}):

        super().__init__(config)
        self._use_fixed_pos = config.get("use_fixed_pos", False)
        self._generate_coin_every_n_steps = config.get(
            "generate_coin_every_n_steps", False
        )
        self._bootstrap_training = config.get("bootstrap_training", False)
        self._last_generation = None
        self.OBSERVATION_SPACE_ = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, 7),
            dtype="uint8",
        )
        self._support_ray_1_12_0()

    @override(CoinGame)
    def _generate_observation(self):

        obs = np.zeros((self.grid_size, self.grid_size, 7), dtype=np.uint8)
        obs[self.red_pos[0], self.red_pos[1], 0] = 1
        obs[self.blue_pos[0], self.blue_pos[1], 1] = 1
        # 2th feature is for the give in coin
        obs[self.coin_pos_give_in[0], self.coin_pos_give_in[1], 2] = 1
        # 3rd feature is for not give in coin
        obs[self.coin_pos_not_give_in[0], self.coin_pos_not_give_in[1], 3] = 1
        # 4th feature is for the threat in coin
        obs[self.coin_pos_threat[0], self.coin_pos_threat[1], 4] = 1
        # if self._add_surrogate_coin:
        # 4th feature is for the surrogate coin
        obs[self.coin_pos_surrogate[0], self.coin_pos_surrogate[1], 5] = 1
        # 6rd feature is for not give in coin
        obs[self.coin_pos_no_threat[0], self.coin_pos_no_threat[1], 6] = 1

        obs = self._apply_optional_invariance_to_the_player_trained(obs)
        return obs

    def _wt_coin_pos_different_from_players_and_other_coin(self):
        if self._use_fixed_pos and self._last_generation is not None:
            self.red_pos = self._last_generation["red_pos"]
            self.blue_pos = self._last_generation["blue_pos"]
            self.coin_pos_give_in = self._last_generation["coin_pos_give_in"]
            self.coin_pos_not_give_in = self._last_generation["coin_pos_not_give_in"]
            self.coin_pos_threat = self._last_generation["coin_pos_threat"]
            self.coin_pos_no_threat = self._last_generation["coin_pos_no_threat"]
            self.coin_pos_surrogate = self._last_generation["coin_pos_surrogate"]
            return None

        success = 0
        while success < self.NUM_AGENTS:
            self.coin_pos_give_in = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos_give_in)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos_give_in)

        success = 0
        while success < self.NUM_AGENTS + 1:
            self.coin_pos_not_give_in = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos_not_give_in)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos_not_give_in)
            success += 1 - self._same_pos(
                self.coin_pos_give_in, self.coin_pos_not_give_in
            )

        success = 0
        while success < self.NUM_AGENTS + 2:
            self.coin_pos_threat = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos_threat)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos_threat)
            success += 1 - self._same_pos(self.coin_pos_give_in, self.coin_pos_threat)
            success += 1 - self._same_pos(
                self.coin_pos_not_give_in, self.coin_pos_threat
            )

        success = 0
        while success < self.NUM_AGENTS + 3:
            self.coin_pos_no_threat = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos_no_threat)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos_no_threat)
            success += 1 - self._same_pos(
                self.coin_pos_give_in, self.coin_pos_no_threat
            )
            success += 1 - self._same_pos(
                self.coin_pos_not_give_in, self.coin_pos_no_threat
            )
            success += 1 - self._same_pos(self.coin_pos_threat, self.coin_pos_no_threat)

        # if self._add_surrogate_coin:
        success = 0
        while success < self.NUM_AGENTS + 4:
            self.coin_pos_surrogate = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos_surrogate)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos_surrogate)
            success += 1 - self._same_pos(
                self.coin_pos_give_in, self.coin_pos_surrogate
            )
            success += 1 - self._same_pos(
                self.coin_pos_not_give_in, self.coin_pos_surrogate
            )
            success += 1 - self._same_pos(self.coin_pos_threat, self.coin_pos_surrogate)
            success += 1 - self._same_pos(
                self.coin_pos_no_threat, self.coin_pos_surrogate
            )

        self._last_generation = {
            "red_pos": self.red_pos,
            "blue_pos": self.blue_pos,
            "coin_pos_give_in": self.coin_pos_give_in,
            "coin_pos_not_give_in": self.coin_pos_not_give_in,
            "coin_pos_threat": self.coin_pos_threat,
            "coin_pos_no_threat": self.coin_pos_no_threat,
            "coin_pos_surrogate": self.coin_pos_surrogate,
        }

    def _compute_reward(self):

        reward_threatener, surrogate_reward_threatener = 0.0, 0.0
        reward_target, surrogate_reward_target = 0.0, 0.0
        generate_new_coins = False
        cell_0_0 = False
        cell_0_1 = False
        cell_0_2 = False
        cell_1_0 = False
        cell_1_1 = False
        cell_1_2 = False
        cell_none = False
        target_alone = False
        threatener_alone = False

        if self.step_count_in_current_episode % self._reward_every_n_steps == 0:

            if self._same_pos(self.red_pos, self.coin_pos_threat):
                action_threatener = 0
            elif (
                self._same_pos(self.red_pos, self.coin_pos_surrogate)
                and self._add_surrogate_coin
            ):
                action_threatener = 1
            elif self._same_pos(self.red_pos, self.coin_pos_no_threat):
                action_threatener = 2
            else:
                action_threatener = None

            if self._same_pos(self.blue_pos, self.coin_pos_give_in):
                action_target = 0
            elif self._same_pos(self.blue_pos, self.coin_pos_not_give_in):
                action_target = 1
            else:
                action_target = None

            if action_threatener is not None and action_target is not None:
                generate_new_coins = True
                reward_target, reward_threatener = self._threat_game_payoff_matrix[
                    action_target, action_threatener, :
                ].tolist()
            else:
                reward_target, reward_threatener = np.min(
                    np.min(self._threat_game_payoff_matrix, axis=1), axis=0
                )

            if action_target == 0:
                if action_threatener == 0:
                    cell_0_0 = True
                elif action_threatener == 1:
                    cell_0_1 = True
                elif action_threatener == 2:
                    cell_0_2 = True
                else:
                    cell_none = True
            elif action_target == 1:
                if action_threatener == 0:
                    cell_1_0 = True
                elif action_threatener == 1:
                    cell_1_1 = True
                elif action_threatener == 2:
                    cell_1_2 = True
                else:
                    cell_none = True
            else:
                cell_none = True

            if self._bootstrap_training:
                bonus = 5.0 / 10.0
                if action_threatener is not None:
                    reward_threatener += bonus / 3.0
                    if cell_none:
                        threatener_alone = True
                if action_target is not None:
                    reward_target += bonus / 2.0
                    if cell_none:
                        target_alone = True

            if self.output_additional_info:
                self._accumulate_info(
                    threatener_alone=threatener_alone,
                    target_alone=target_alone,
                    cell_0_0=cell_0_0,
                    cell_0_1=cell_0_1,
                    cell_0_2=cell_0_2,
                    cell_1_0=cell_1_0,
                    cell_1_1=cell_1_1,
                    cell_1_2=cell_1_2,
                    cell_none=cell_none,
                )

        reward_list = [reward_threatener, reward_target]

        self.reward_list = reward_list
        self.surrogate_reward_list = reward_list

        if (
            self._generate_coin_every_n_steps
            and self.step_count_in_current_episode % self._generate_coin_every_n_steps
            == 0
        ):
            generate_new_coins = True

        return reward_list, generate_new_coins


class PerfectCoordMixing:
    NUM_ACTIONS = 6
    ACTION_SPACE_ = Discrete(NUM_ACTIONS)
    MOVES = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([0, 0]),
        None,
    ]

    def _move_players(self, actions):
        "Action number 6 is picking a random action."

        action_red, action_blue = actions

        if action_red == 5:
            action_red = random.randint(0, 4)
        if action_blue == 5:
            action_blue = random.randint(0, 4)

        super()._move_players([action_red, action_blue])


class PerfectCoordChickenCoinGame(PerfectCoordMixing, ChickenCoinGame):
    NAME = "PerfectCoordChickenCoinGame"


class PerfectCoordThreatCoinGame(PerfectCoordMixing, ThreatCoinGame):
    NAME = "PerfectCoordChickenCoinGame"


class PerfectCoordThreatCoinGameV2(PerfectCoordMixing, ThreatCoinGameV2):
    NAME = "PerfectCoordChickenCoinGame"
