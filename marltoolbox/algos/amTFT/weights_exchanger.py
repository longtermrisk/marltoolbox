import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from marltoolbox.algos.amTFT.base import (
    OPP_SELFISH_POLICY_IDX,
    OPP_COOP_POLICY_IDX,
    OWN_SELFISH_POLICY_IDX,
    OWN_COOP_POLICY_IDX,
    AmTFTReferenceClass,
    WORKING_STATES_IN_EVALUATION,
)

logger = logging.getLogger(__name__)


class WeightsExchanger:
    """
    Mixin to add the method on_train_result (called by the callback of the
    same name).
    For both amTFT policies: copy the weights from the opponent policy into the
    policy.
    Copy policy OWN_COOP_POLICY_IDX from opponent into
    OPP_COOP_POLICY_IDX inside own policy.
    Copy OWN_SELFISH_POLICY_IDX from opponent into
    OPP_SELFISH_POLICY_IDX inside own policy.
    """

    @staticmethod
    def on_train_result(trainer, *args, **kwargs):
        WeightsExchanger._share_weights_during_training(trainer)

    @staticmethod
    def _share_weights_during_training(trainer):
        local_policy_map = trainer.workers.local_worker().policy_map
        policy_ids = list(local_policy_map.keys())
        assert len(policy_ids) == 2, (
            "amTFT only works in two player " "environments"
        )

        in_training = WeightsExchanger._are_policies_in_training(
            local_policy_map
        )
        if in_training:
            WeightsExchanger._check_only_amTFT_policies(local_policy_map)
            policies_weights = trainer.get_weights()
            policies_weights = (
                WeightsExchanger._get_opp_policies_from_opponents(
                    policy_ids, local_policy_map, policies_weights
                )
            )
            trainer.set_weights(policies_weights)

    @staticmethod
    def _are_policies_in_training(local_policy_map):
        in_training = all(
            [
                isinstance(policy, AmTFTReferenceClass)
                and policy.working_state not in WORKING_STATES_IN_EVALUATION
                for policy in local_policy_map.values()
            ]
        )
        return in_training

    @staticmethod
    def _get_opp_policies_from_opponents(
        policy_ids, local_policy_map, policies_weights
    ):

        for i, policy_id in enumerate(policy_ids):
            opp_policy_id = policy_ids[(i + 1) % 2]
            policy = local_policy_map[policy_id]

            policies_weights = WeightsExchanger._load_weights_of_the_opponent(
                policy, policies_weights, policy_id, opp_policy_id
            )
        return policies_weights

    @staticmethod
    def _load_weights_of_the_opponent(
        policy, policies_weights, policy_id, opp_policy_id
    ):

        opp_coop_pi_key = policy.nested_key(OPP_COOP_POLICY_IDX)
        own_coop_pi_key = policy.nested_key(OWN_COOP_POLICY_IDX)
        opp_selfish_pi_key = policy.nested_key(OPP_SELFISH_POLICY_IDX)
        own_selfish_pi_key = policy.nested_key(OWN_SELFISH_POLICY_IDX)

        # share weights during training of amTFT
        policies_weights[policy_id][opp_coop_pi_key] = policies_weights[
            opp_policy_id
        ][own_coop_pi_key]
        policies_weights[policy_id][opp_selfish_pi_key] = policies_weights[
            opp_policy_id
        ][own_selfish_pi_key]
        return policies_weights

    @staticmethod
    def _check_only_amTFT_policies(local_policy_map):
        for policy in local_policy_map.values():
            assert isinstance(policy, AmTFTReferenceClass), (
                "if amTFT is training then "
                "all players must be "
                "using amTFT too"
            )
