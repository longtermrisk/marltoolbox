import itertools

import numpy as np
from ray import rllib


# DEFAULT_CONFIG

class WelfareCoordinationTorchPolicy(rllib.policy.TorchPolicy):
    # def __init__(self):

    OWN_PLAYER_IDX = 0
    OPP_PLAYER_IDX = 1

    def setup_meta_game(self, all_welfare_pairs_wt_payoffs: dict,
                        all_trained_policies):
        """
        :param all_welfare_pairs_wt_payoffs: Dictionnary containing as keys,
            welfare function names for each player separated by "-".
            And as value the tuple of payoff for each player.
        """
        self.all_welfare_pairs_wt_payoffs = all_welfare_pairs_wt_payoffs
        self._list_all_welfares_fn()
        self._list_all_set_of_welfare_fn()

    def _list_all_welfares_fn(self):
        all_welfare_fn = []
        for welfare_pairs_key in self.all_welfare_pairs_wt_payoffs.keys():
            for welfare_fn in self._key_to_pair_of_welfare_names(
                    welfare_pairs_key):
                all_welfare_fn.append(welfare_fn)
        self.all_welfare_fn = tuple(set(all_welfare_fn))

    @staticmethod
    def _key_to_pair_of_welfare_names(key):
        return key.split("-")

    def _list_all_set_of_welfare_fn(self):
        """
        Each conbinaison is a potentiel action in the meta game
        """
        welfare_fn_sets = []
        for n_items in range(1, len(self.all_welfare_fn) + 1):
            combinations_object = itertools.combinations(
                self.all_welfare_fn, n_items)
            combinations_object = list(combinations_object)
            combinations_set = [set(combi) for combi in combinations_object]
            welfare_fn_sets.extend(combinations_set)
        self.welfare_fn_sets = tuple(set(welfare_fn_sets))

    def solve_meta_game(self, tau):
        """
        solve the game by finding which welfare set to annonce
        :param tau:
        """
        self.tau = tau
        self.selected_pure_policy_idx = None
        self.best_objective = - np.inf

        self._search_for_best_action()

    def _search_for_best_action(self):
        for idx, welfare_set_annonced in enumerate(self.welfare_fn_sets):
            optimization_objective = \
                self._compute_optimization_objective(self.tau,
                                                     welfare_set_annonced)
            self._keep_action_if_best(optimization_objective, idx)

    def _compute_optimization_objective(
            self, tau, welfare_set_annonced):

        self._compute_payoffs_for_every_opponent_action(welfare_set_annonced)

        opp_best_response_idx = self._get_opp_best_response_idx()
        exploitability_term = \
            tau * self._get_own_payoff(opp_best_response_idx)

        payoffs = self._get_all_possible_payoffs_excluding_one(
            excluding_idx=opp_best_response_idx)
        robustness_term = (1 - tau) * sum(payoffs) / len(payoffs)

        return exploitability_term + robustness_term

    def _compute_payoffs_for_every_opponent_action(self, welfare_set_annonced):
        self.all_possible_payoffs = []
        # For every opponent actions
        for welfare_set_idx, opp_wefare_set in enumerate(self.welfare_fn_sets):
            payoff = self._compute_meta_payoff(
                welfare_set_annonced, opp_wefare_set)
            self.all_possible_payoffs.append(payoff)

    def _compute_meta_payoff(self, own_welfare_set, opp_welfare_set):
        welfare_fn_intersection = own_welfare_set & opp_welfare_set

        if len(welfare_fn_intersection) == 0:
            return self._get_payoff_for_default_strategies()
        else:
            return self._get_payoff_averaging_over_all_join_strategies(
                welfare_fn_intersection)

    def _get_payoff_for_default_strategies(self):
        return self._read_payoff_from_data(
            self.own_default_welfare_fn, self.opp_default_welfare_fn)

    def _get_payoff_averaging_over_all_join_strategies(
            self, welfare_fn_intersection):
        payoffs = []
        for welfare_fn in welfare_fn_intersection:
            payoffs.append(self._read_payoff_from_data(
                welfare_fn, welfare_fn))
        mean_payoff = sum(payoffs) / len(payoffs)
        return mean_payoff

    def _read_payoff_from_data(self, own_welfare, opp_welfare):
        welfare_pair_name = self._from_pair_of_welfare_names_to_key(
            own_welfare, opp_welfare
        )
        return self.all_welfare_pairs_wt_payoffs[welfare_pair_name]

    @staticmethod
    def _from_pair_of_welfare_names_to_key(own_welfare_set, opp_welfare_set):
        return f"{own_welfare_set}-{opp_welfare_set}"

    def _get_opp_best_response_idx(self):
        opp_payoffs = [payoffs[self.OPP_PLAYER_IDX]
                       for payoffs in self.all_possible_payoffs]
        return opp_payoffs.index(max(opp_payoffs))

        # opp_best_response_idx = None
        # opp_best_response_value = - np.inf
        # for idx, (own_payoffs, opp_payoffs) in enumerate(
        #         self.all_possible_payoffs):
        #     if opp_payoffs > opp_best_response_value:
        #         opp_best_response_value = opp_payoffs
        #         opp_best_response_idx = idx
        # return opp_best_response_idx

    def _get_all_possible_payoffs_excluding_one(self, excluding_idx):
        own_payoffs = []
        for welfare_set_idx, _ in enumerate(self.welfare_fn_sets):
            if welfare_set_idx != excluding_idx:
                own_payoff = self._get_own_payoff(welfare_set_idx)
                own_payoffs.append(own_payoff)
        assert len(own_payoffs) == len(self.welfare_fn_sets) - 1
        return own_payoffs

    def _get_own_payoff(self, idx):
        return self._get_payoff(player_idx=self.OWN_PLAYER_IDX,
                                joint_strategy_idx=idx)

    def _get_opp_payoff(self, idx):
        return self._get_payoff(player_idx=self.OPP_PLAYER_IDX,
                                joint_strategy_idx=idx)

    def _get_payoff(self, player_idx, joint_strategy_idx):
        return self.all_possible_payoffs[joint_strategy_idx][player_idx]

    def _keep_action_if_best(self, optimization_objective, idx):
        if optimization_objective > self.best_objective:
            self.best_objective = optimization_objective
            self.selected_pure_policy_idx = idx
