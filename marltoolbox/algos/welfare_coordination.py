import itertools
import logging
import random

import torch
import numpy as np
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override

from marltoolbox.algos import population, amTFT

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = merge_dicts(
    population.DEFAULT_CONFIG,
    {
        "nested_policies": [
            # You need to provide the policy class for every nested Policies
            {
                "Policy_class": None,
                "config_update": {},
            },
        ],
        "solve_meta_game_after_init": True,
        "tau": None,
        "all_welfare_pairs_wt_payoffs": None,
        "own_player_idx": None,
        "opp_player_idx": None,
        "own_default_welfare_fn": None,
        "opp_default_welfare_fn": None,
        "distrib_over_welfare_sets_to_annonce": False,
    },
)


class MetaGameSolver:
    def setup_meta_game(
        self,
        all_welfare_pairs_wt_payoffs: dict,
        own_player_idx: int,
        opp_player_idx: int,
        own_default_welfare_fn: str,
        opp_default_welfare_fn: str,
    ):
        """

        :param all_welfare_pairs_wt_payoffs: Dictionnary containing as keys,
            welfare function names for each player separated by "-".
            And as value the tuple of payoff for each player.
        :param own_player_idx: idx of own policy's payoff as stored in
            all_welfare_pairs_wt_payoffs
        :param opp_player_idx: idx of opponent policy's payoff as stored in
            all_welfare_pairs_wt_payoffs
        :param own_default_welfare_fn: own default welfare function to use
            when the intersection between the sets announced is null
        :param opp_default_welfare_fn: opponent default welfare function to
            use when the intersection between the sets announced is null
        :return:
        """
        self.all_welfare_pairs_wt_payoffs = all_welfare_pairs_wt_payoffs
        self.own_player_idx = own_player_idx
        self.opp_player_idx = opp_player_idx
        self.own_default_welfare_fn = own_default_welfare_fn
        self.opp_default_welfare_fn = opp_default_welfare_fn
        self.all_welfare_fn = MetaGameSolver.list_all_welfares_fn(
            self.all_welfare_pairs_wt_payoffs
        )
        self._list_all_set_of_welfare_fn()

    @staticmethod
    def list_all_welfares_fn(all_welfare_pairs_wt_payoffs):
        """List all welfare functions in the given data structure"""
        all_welfare_fn = []
        for welfare_pairs_key in all_welfare_pairs_wt_payoffs.keys():
            for welfare_fn in MetaGameSolver.key_to_pair_of_welfare_names(
                welfare_pairs_key
            ):
                all_welfare_fn.append(welfare_fn)
        return sorted(tuple(set(all_welfare_fn)))

    @staticmethod
    def key_to_pair_of_welfare_names(key):
        """Convert a key to the two welfare functions composing this key"""
        return key.split("-")

    def _list_all_set_of_welfare_fn(self):
        """
        Each conbinaison is a potentiel action in the meta game
        """
        welfare_fn_sets = []
        for n_items in range(1, len(self.all_welfare_fn) + 1):
            combinations_object = itertools.combinations(
                self.all_welfare_fn, n_items
            )
            combinations_object = list(combinations_object)
            combinations_set = [
                frozenset(combi) for combi in combinations_object
            ]
            welfare_fn_sets.extend(combinations_set)
        self.welfare_fn_sets = sorted(tuple(set(welfare_fn_sets)))

    def solve_meta_game(self, tau):
        """
        solve the game by finding which welfare set to annonce
        :param tau:
        """
        print("================================================")
        print(
            f"Player (idx={self.own_player_idx}) starts solving meta game "
            f"with tau={tau}"
        )
        self.tau = tau
        self.selected_pure_policy_idx = None
        self.best_objective = -np.inf

        self._search_for_best_action()
        print(
            "===> after solving meta game: self.selected_pure_policy_idx",
            self.selected_pure_policy_idx,
            "==>",
            self.welfare_fn_sets[self.selected_pure_policy_idx],
        )
        # TODO change that
        self.welfare_set_to_annonce = self.welfare_fn_sets[
            self.selected_pure_policy_idx
        ]

    def _search_for_best_action(self):
        self.tau_log = []
        for idx, welfare_set_annonced in enumerate(self.welfare_fn_sets):
            optimization_objective = self._compute_optimization_objective(
                self.tau, welfare_set_annonced
            )
            self._keep_action_if_best(optimization_objective, idx)

    def _compute_optimization_objective(self, tau, welfare_set_annonced):
        self._compute_payoffs_for_every_opponent_action(welfare_set_annonced)

        opp_best_response_idx = self._get_opp_best_response_idx()

        exploitability_term = tau * self._get_own_payoff(opp_best_response_idx)
        payoffs = self._get_all_possible_payoffs_excluding_provided(
            # excluding_idx=opp_best_response_idx
        )
        robustness_term = (1 - tau) * sum(payoffs) / len(payoffs)

        optimization_objective = exploitability_term + robustness_term

        print(f"========")
        print(f"compute objective for set {welfare_set_annonced}")
        print(
            f"opponent best response is {opp_best_response_idx} => "
            f"{self.welfare_fn_sets[opp_best_response_idx]}"
        )
        print("exploitability_term", exploitability_term)
        print("robustness_term", robustness_term)
        print("optimization_objective", optimization_objective)

        self.tau_log.append(
            {
                "welfare_set_annonced": welfare_set_annonced,
                "opp_best_response_idx": opp_best_response_idx,
                "opp_best_response_set": self.welfare_fn_sets[
                    opp_best_response_idx
                ],
                "robustness_term": robustness_term,
                "optimization_objective": optimization_objective,
            }
        )

        return optimization_objective

    def _compute_payoffs_for_every_opponent_action(self, welfare_set_annonced):
        self.all_possible_payoffs = []
        # For every opponent actions
        for welfare_set_idx, opp_wefare_set in enumerate(self.welfare_fn_sets):
            payoff = self._compute_meta_payoff(
                welfare_set_annonced, opp_wefare_set
            )
            self.all_possible_payoffs.append(payoff)

    def _compute_meta_payoff(self, own_welfare_set, opp_welfare_set):
        welfare_fn_intersection = own_welfare_set & opp_welfare_set

        if len(welfare_fn_intersection) == 0:
            return self._read_payoffs_for_default_strategies()
        else:
            return self._read_payoffs_and_average_over_welfare_intersection(
                welfare_fn_intersection
            )

    def _read_payoffs_for_default_strategies(self):
        return self._read_own_payoff_from_data(
            self.own_default_welfare_fn, self.opp_default_welfare_fn
        )

    def _read_payoffs_and_average_over_welfare_intersection(
        self, welfare_fn_intersection
    ):
        payoffs_player_1 = []
        payoffs_player_2 = []
        for welfare_fn in welfare_fn_intersection:
            payoff_player_1, payoff_player_2 = self._read_own_payoff_from_data(
                welfare_fn, welfare_fn
            )
            payoffs_player_1.append(payoff_player_1)
            payoffs_player_2.append(payoff_player_2)
        mean_payoff_p1 = sum(payoffs_player_1) / len(payoffs_player_1)
        mean_payoff_p2 = sum(payoffs_player_2) / len(payoffs_player_2)
        return mean_payoff_p1, mean_payoff_p2

    def _read_own_payoff_from_data(self, own_welfare, opp_welfare):
        welfare_pair_name = self.from_pair_of_welfare_names_to_key(
            own_welfare, opp_welfare
        )
        return self.all_welfare_pairs_wt_payoffs[welfare_pair_name]

    @staticmethod
    def from_pair_of_welfare_names_to_key(own_welfare_set, opp_welfare_set):
        """Convert two welfare functions into a key identifying this pair"""
        return f"{own_welfare_set}-{opp_welfare_set}"

    def _get_opp_best_response_idx(self):
        opp_payoffs = [
            self._get_opp_payoff(i)
            for i in range(len(self.all_possible_payoffs))
        ]
        return opp_payoffs.index(max(opp_payoffs))

    def _get_all_possible_payoffs_excluding_provided(self, excluding_idx=()):
        own_payoffs = []
        for welfare_set_idx, _ in enumerate(self.welfare_fn_sets):
            if welfare_set_idx not in excluding_idx:
                own_payoff = self._get_own_payoff(welfare_set_idx)
                own_payoffs.append(own_payoff)
        assert len(own_payoffs) == len(self.welfare_fn_sets) - len(
            excluding_idx
        )
        return own_payoffs

    def _get_own_payoff(self, idx):
        return self._get_payoff(
            player_idx=self.own_player_idx, joint_strategy_idx=idx
        )

    def _get_opp_payoff(self, idx):
        return self._get_payoff(
            player_idx=self.opp_player_idx, joint_strategy_idx=idx
        )

    def _get_payoff(self, player_idx, joint_strategy_idx):
        return self.all_possible_payoffs[joint_strategy_idx][player_idx]

    def _keep_action_if_best(self, optimization_objective, idx):
        if optimization_objective > self.best_objective:
            self.best_objective = optimization_objective
            self.selected_pure_policy_idx = idx


class WelfareCoordinationTorchPolicy(
    population.PopulationOfIdenticalAlgo,
    MetaGameSolver,
    amTFT.AmTFTReferenceClass,
):
    def __init__(
        self,
        observation_space,
        action_space,
        config,
        after_init_nested=None,
        **kwargs,
    ):
        self.policy_ckpts_per_welfare = config["policy_checkpoints"]
        self._welfare_in_use = None

        super().__init__(
            observation_space,
            action_space,
            config,
            after_init_nested=after_init_nested,
            **kwargs,
        )

        assert (
            self._use_distribution_over_sets()
            or self.config["solve_meta_game_after_init"]
        )
        if self.config["solve_meta_game_after_init"]:
            assert not self.config["distrib_over_welfare_sets_to_annonce"]
            self._choose_which_welfare_set_to_annonce()
        if self._use_distribution_over_sets():
            self._init_distribution_over_sets_to_announce()

        self._welfare_set_annonced = False
        self._welfare_set_in_use = None
        self._welfare_chosen_for_epi = False
        self._intersection_of_welfare_sets = None

    def _init_distribution_over_sets_to_announce(self):
        assert not self.config["solve_meta_game_after_init"]
        self.setup_meta_game(
            all_welfare_pairs_wt_payoffs=self.config[
                "all_welfare_pairs_wt_payoffs"
            ],
            own_player_idx=self.config["own_player_idx"],
            opp_player_idx=self.config["opp_player_idx"],
            own_default_welfare_fn=self.config["own_default_welfare_fn"],
            opp_default_welfare_fn=self.config["opp_default_welfare_fn"],
        )
        self.stochastic_announcement_policy = torch.distributions.Categorical(
            probs=self.config["distrib_over_welfare_sets_to_annonce"]
        )
        self._stochastic_selection_of_welfare_set_to_announce()

    def _stochastic_selection_of_welfare_set_to_announce(self):
        welfare_set_idx_selected = self.stochastic_announcement_policy.sample()
        self.welfare_set_to_annonce = self.welfare_fn_sets[
            welfare_set_idx_selected
        ]
        self._to_log["welfare_set_to_annonce"] = str(
            self.welfare_set_to_annonce
        )
        print("self.welfare_set_to_annonce", self.welfare_set_to_annonce)

    def _choose_which_welfare_set_to_annonce(self):
        self.setup_meta_game(
            all_welfare_pairs_wt_payoffs=self.config[
                "all_welfare_pairs_wt_payoffs"
            ],
            own_player_idx=self.config["own_player_idx"],
            opp_player_idx=self.config["opp_player_idx"],
            own_default_welfare_fn=self.config["own_default_welfare_fn"],
            opp_default_welfare_fn=self.config["opp_default_welfare_fn"],
        )
        self.solve_meta_game(self.config["tau"])
        self._to_log[f"tau_{self.tau}"] = self.tau_log[
            self.selected_pure_policy_idx
        ]

    @property
    def policy_checkpoints(self):
        if self._welfare_in_use is None:
            return None

        return self.policy_ckpts_per_welfare[self._welfare_in_use]

    @policy_checkpoints.setter
    def policy_checkpoints(self, value):
        logger.warning(
            f"ignoring setting self.policy_checkpoints to value {value}"
        )

    @override(population.PopulationOfIdenticalAlgo)
    def set_algo_to_use(self):
        """
        Called by a callback at the start of every episode.
        """
        if self.policy_checkpoints is not None:
            super().set_algo_to_use()

    @override(population.PopulationOfIdenticalAlgo)
    def on_episode_start(
        self,
        *args,
        worker,
        policy,
        policy_id,
        policy_ids,
        **kwargs,
    ):
        if not self._welfare_set_annonced:
            # Called only by one agent, not by both
            self._annonce_welfare_sets(worker)
        if not self._welfare_chosen_for_epi:
            # Called only by one agent, not by both
            self._coordinate_on_welfare_to_use_for_epi(worker)
        super().on_episode_start(
            *args,
            worker,
            policy,
            policy_id,
            policy_ids,
            **kwargs,
        )
        assert self._welfare_set_annonced
        assert self._welfare_chosen_for_epi

    @staticmethod
    def _annonce_welfare_sets(
        worker,
    ):
        intersection_of_welfare_sets = (
            WelfareCoordinationTorchPolicy._find_intersection_of_welfare_sets(
                worker
            )
        )

        for policy_id, policy in worker.policy_map.items():
            if isinstance(policy, WelfareCoordinationTorchPolicy):
                WelfareCoordinationTorchPolicy._inform_policy_of_situation(
                    policy, intersection_of_welfare_sets
                )

    @staticmethod
    def _find_intersection_of_welfare_sets(worker):
        welfare_sets_annonced = [
            policy.welfare_set_to_annonce
            for policy in worker.policy_map.values()
        ]
        assert len(welfare_sets_annonced) == 2
        welfare_set_intersection = (
            welfare_sets_annonced[0] & welfare_sets_annonced[1]
        )
        return welfare_set_intersection

    @staticmethod
    def _inform_policy_of_situation(policy, intersection_of_welfare_sets):
        policy._intersection_of_welfare_sets = intersection_of_welfare_sets
        if len(intersection_of_welfare_sets) == 0:
            policy._welfare_set_in_use = set([policy.own_default_welfare_fn])
        else:
            policy._welfare_set_in_use = intersection_of_welfare_sets

        policy._welfare_set_annonced = True

    @staticmethod
    def _coordinate_on_welfare_to_use_for_epi(worker):
        welfare_to_play = None
        for policy_id, policy in worker.policy_map.items():
            if isinstance(policy, WelfareCoordinationTorchPolicy):
                if len(policy._intersection_of_welfare_sets) > 0:
                    if welfare_to_play is None:
                        welfare_to_play = random.choice(
                            tuple(policy._welfare_set_in_use)
                        )
                    policy._welfare_in_use = welfare_to_play
                    msg = (
                        "Policies coordinated "
                        "to play for the next epi: "
                        f"_welfare_set_in_use={policy._welfare_set_in_use}"
                        f"and selected: policy._welfare_in_use={policy._welfare_in_use}"
                    )
                    logger.info(msg)
                    print(msg)
                else:
                    policy._welfare_in_use = random.choice(
                        tuple(policy._welfare_set_in_use)
                    )
                    msg = (
                        "Policies did NOT coordinate "
                        "to play for the next epi: "
                        f"_welfare_set_in_use={policy._welfare_set_in_use}"
                        f"and selected: policy._welfare_in_use={policy._welfare_in_use}"
                    )
                    logger.info(msg)
                    print(msg)
                policy._welfare_chosen_for_epi = True
                policy._to_log["welfare_in_use"] = policy._welfare_in_use
                policy._to_log[
                    "welfare_set_in_use"
                ] = policy._welfare_set_in_use

    def on_episode_end(
        self,
        *args,
        **kwargs,
    ):
        self._welfare_chosen_for_epi = False
        if hasattr(self.algorithms[self.active_algo_idx], "on_episode_end"):
            self.algorithms[self.active_algo_idx].on_episode_end(
                *args, **kwargs
            )
        if self._use_distribution_over_sets():
            self._stochastic_selection_of_welfare_set_to_announce()
            self._welfare_set_annonced = False

    def _use_distribution_over_sets(self):
        return not self.config["distrib_over_welfare_sets_to_annonce"] is False

    @property
    @override(population.PopulationOfIdenticalAlgo)
    def to_log(self):
        to_log = {
            "meta_policy": self._to_log,
            "nested_policy": {
                f"policy_{algo_idx}": algo.to_log
                for algo_idx, algo in enumerate(self.algorithms)
                if hasattr(algo, "to_log")
            },
        }
        return to_log

    @to_log.setter
    @override(population.PopulationOfIdenticalAlgo)
    def to_log(self, value):
        if value == {}:
            for algo in self.algorithms:
                if hasattr(algo, "to_log"):
                    algo.to_log = {}

        self._to_log = value
