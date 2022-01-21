import numpy as np
import pyspiel

from marltoolbox.envs.simple_bargaining import SimpleBargaining

RLLIB_SIMPLE_BARGAINING = SimpleBargaining({})

_NUM_PLAYERS = 2
_N_DISCRETE = 11
_N_ACTIONS = 2
_GAME_TYPE = pyspiel.GameType(
    short_name="python_simple_bargaining",
    long_name="Python Simple Bargaining",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.ONE_SHOT,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_N_ACTIONS * _N_DISCRETE,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=0.0,
    max_utility=3.0,
    utility_sum=0.0,
    max_game_length=1,
)


class SimpleBargainingGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return SimpleBargainingState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return SimpleBargainingObserver(
            iig_obs_type or pyspiel.IIGObservationType(), params
        )


class SimpleBargainingState(pyspiel.State):
    G = GAINS_FROM_TRADE_FACTOR = 3.0
    MULTIPLIER = 0.2
    PL0_T0, PL0_T1, PL1_T0, PL1_T1 = np.array([3, 9, 7, 2]) * MULTIPLIER

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._game_turn = 0
        self._actions_possible = list(range(_N_ACTIONS * _N_DISCRETE))
        self._game_over = False
        self._next_player = 0
        self.action_player_0 = [None] * _N_ACTIONS
        self.action_player_1 = [None] * _N_ACTIONS
        self.rllib_game = RLLIB_SIMPLE_BARGAINING
        self.verbose = False
        if self.verbose:
            print("start epi")

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every sequential-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        else:
            return self._next_player

    def legal_actions(self, player=None):
        """Returns a list of legal actions, sorted in ascending order."""
        legal_actions = None
        if self._game_turn == 4 or self._game_over:
            legal_actions = []
        elif player is not None and player != self._next_player:
            legal_actions = []
        elif self._game_turn == 0 or self._game_turn == 2:
            legal_actions = self._actions_possible[:_N_DISCRETE]
        elif self._game_turn == 1 or self._game_turn == 3:
            legal_actions = self._actions_possible[_N_DISCRETE:]
        if self.verbose:
            print(self._game_turn, "legal_actions", legal_actions)
        return legal_actions

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        raise NotImplementedError()

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        self._assert_turn_action(self._game_turn, action)
        if self._game_turn == 0:
            self.action_player_0[0] = self._get_float_values(action)
            self._next_player = 0
        elif self._game_turn == 1:
            self.action_player_0[1] = self._get_float_values(action)
            self._next_player = 1
        elif self._game_turn == 2:
            self.action_player_1[0] = self._get_float_values(action)
            self._next_player = 1
        elif self._game_turn == 3:
            self.action_player_1[1] = self._get_float_values(action)
            self._game_over = True
            self._next_player = 0
        else:
            raise ValueError(f"self._game_over {self._game_over}")
        if self.verbose:
            print(self._game_turn, "_apply_action", action)
        self._game_turn += 1

    @staticmethod
    def _assert_turn_action(i, action):
        i = i % _N_ACTIONS
        assert (
            (_N_DISCRETE * i) <= action <= (_N_DISCRETE * (i + 1))
        ), f"action {action}, _N_DISCRETE * i: {_N_DISCRETE * i}"

    @staticmethod
    def _get_float_values(action):
        return (action % _N_DISCRETE) / _N_DISCRETE

    def _action_to_string(self, player, action):
        """Action -> string."""
        return f"{player}_{action}"

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if not self._game_over:
            return [0.0, 0.0]

        rewards = self.rllib_game._get_players_rewards(
            self.action_player_0, self.action_player_1
        )

        if self.verbose:
            print(self._game_turn, "returns", rewards)

        return rewards

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return f"{self.action_player_0}_{self.action_player_1}"


class SimpleBargainingObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(
                f"Observation parameters not supported; passed {params}"
            )

        shape = (_N_DISCRETE,)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        obs = self.dict["observation"]
        obs.fill(0)
        if state._game_turn == 1:
            if player == 0:
                pl0_act0 = self._get_action_idx(state.action_player_0[0])
                obs[pl0_act0] = 1
        elif state._game_turn == 3:
            if player == 1:
                pl1_act0 = self._get_action_idx(state.action_player_1[0])
                obs[pl1_act0] = 1
        elif (
            state._game_turn == 0
            or state._game_turn == 2
            or state._game_turn == 4
        ):
            pass
        else:
            raise ValueError(f"state._game_turn {state._game_turn}")
        assert np.all(self.dict["observation"] == obs)

    @staticmethod
    def _get_action_idx(value):
        return int(value * _N_DISCRETE)

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        pieces.extend(state.action_player_0)
        pieces.extend(state.action_player_1)
        return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, SimpleBargainingGame)
