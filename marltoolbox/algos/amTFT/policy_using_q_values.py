from marltoolbox.algos.amTFT.base import (
    OPP_SELFISH_POLICY_IDX,
    OWN_SELFISH_POLICY_IDX,
)
from marltoolbox.algos.amTFT.base_policy import AmTFTPolicyBase


class amTFTQValuesTorchPolicy(AmTFTPolicyBase):
    def __init__(self, *args, **kwargs):
        assert (
            0
        ), "The amTFTQValuesTorchPolicy algorithm is not entirely functional."
        super().__init__(*args, **kwargs)

    def _compute_debit(
        self,
        last_obs,
        opp_action,
        worker,
        base_env,
        episode,
        env_index,
        coop_opp_simulated_action,
    ):
        approximated_debit = self._compute_debit_from_q_values(
            last_obs, opp_action, coop_opp_simulated_action
        )
        return approximated_debit

    def _compute_debit_from_q_values(
        self, last_obs, opp_action, coop_opp_simulated_action
    ):
        self.selfish_opp_q_values = self._compute_selfish_opp_q_values(
            last_obs
        )

        # TODO solve problem with Q-value changed by exploration => check if this is done
        temperature_used_for_exploration = self.algorithms[
            OWN_SELFISH_POLICY_IDX
        ].exploration.temperature

        # TODO Here we are evaluating the gains for the opponent but we are using the selfish q values trained
        #   between two agents, thus we can't really see any gain from exploitation
        debit = (
            self.selfish_opp_q_values[0, opp_action]
            * temperature_used_for_exploration
            - self.selfish_opp_q_values[0, coop_opp_simulated_action]
            * temperature_used_for_exploration
        )
        self._to_log["raw_debit"] = debit
        return debit

    def _compute_selfish_opp_q_values(self, last_obs):
        _, _, selfish_opp_extra_fetches = self.algorithms[
            OPP_SELFISH_POLICY_IDX
        ].compute_actions([last_obs])
        selfish_opp_q_values = selfish_opp_extra_fetches["q_values"]
        assert len(selfish_opp_q_values) == 1, "batch size need to be 1"
        return selfish_opp_q_values

    def _compute_punishment_duration(
        self, opp_action, coop_opp_simulated_action, worker, last_obs
    ):
        return self._compute_punishment_duration_from_q_values(
            opp_action, coop_opp_simulated_action
        )

    def _compute_punishment_duration_from_q_values(
        self, opp_action, coop_opp_simulated_action
    ):
        q_coop = self.coop_opp_extra_fetches["q_values"]
        q_selfish = self.selfish_opp_q_values

        # TODO This is only going to work if each action has the same impact on the reward
        # TODO solve problem with Q-value changed by exploration (temperature) => check if this is done here

        print("q_coop", q_coop)
        print("q_selfish", q_selfish)
        # TODO This is not completely fine. I should use the "Q values of the opponent reward while cooperating"
        #   but I am using the "Q values of both agents welfare while cooperating".
        #   This is why there is "/2" to balance that a bit. It would be better to train another Q-network
        #   which would only be used here (not to produce any action) to access the right Q values.
        opp_expected_lost_per_step = (
            q_coop[coop_opp_simulated_action] / 2 - q_selfish[opp_action]
        )
        print("self.total_debit", self.total_debit)
        print("opp_expected_lost_per_step", opp_expected_lost_per_step)
        n_steps_equivalent = (
            self.total_debit * self.punishment_multiplier
        ) / opp_expected_lost_per_step
        return int(n_steps_equivalent + 1 - 1e-6)
