import copy
import random
from typing import Dict, List

from ray import tune


def mix_policies_in_given_rllib_configs(
    all_rllib_configs: List[Dict], n_mix_per_config: int
) -> dict:
    """
    Mix the policies of a list of RLLib config dictionaries. Limited to
    RLLib config with 2 policies. (Not used by the SelfAndCrossPlayEvaluator)


    :param all_rllib_configs: all rllib config
    :param n_mix_per_config: number of mix to create for each rllib config
        provided
    :return: a single rllib config with a grid search over all the mixed
        pair of policies
    """
    assert (
        n_mix_per_config <= len(all_rllib_configs) - 1
    ), f" {n_mix_per_config} <= {len(all_rllib_configs) - 1}"
    policy_ids = all_rllib_configs[0]["multiagent"]["policies"].keys()
    assert len(policy_ids) == 2, (
        "only supporting config dict with 2 RLLib " "policies"
    )
    _assert_all_config_use_the_same_policies(all_rllib_configs, policy_ids)

    policy_config_variants = _gather_policy_variant_per_policy_id(
        all_rllib_configs, policy_ids
    )

    master_config = _create_one_master_config(
        all_rllib_configs, policy_config_variants, policy_ids, n_mix_per_config
    )
    return master_config


def _create_one_master_config(
    all_rllib_configs, policy_config_variants, policy_ids, n_mix_per_config
):
    all_policy_mix = []
    player_1, player_2 = policy_ids
    for config_idx, p1_policy_config in enumerate(
        policy_config_variants[player_1]
    ):
        policies_mixes = _produce_n_mix_with_player_2_policies(
            policy_config_variants,
            player_2,
            config_idx,
            n_mix_per_config,
            player_1,
            p1_policy_config,
        )
        all_policy_mix.extend(policies_mixes)

    master_config = copy.deepcopy(all_rllib_configs[0])
    print("len(all_policy_mix)", len(all_policy_mix))
    master_config["multiagent"]["policies"] = tune.grid_search(all_policy_mix)
    return master_config


def _produce_n_mix_with_player_2_policies(
    policy_config_variants,
    player_2,
    config_idx,
    n_mix_per_config,
    player_1,
    p1_policy_config,
):
    p2_policy_configs_sampled = _get_p2_policies_samples_excluding_self(
        policy_config_variants, player_2, config_idx, n_mix_per_config
    )
    policies_mixes = []
    for p2_policy_config in p2_policy_configs_sampled:
        policy_mix = {
            player_1: p1_policy_config,
            player_2: p2_policy_config,
        }
        policies_mixes.append(policy_mix)
    return policies_mixes


def _get_p2_policies_samples_excluding_self(
    policy_config_variants, player_2, config_idx, n_mix_per_config
):
    p2_policy_config_variants = copy.deepcopy(policy_config_variants[player_2])
    p2_policy_config_variants.pop(config_idx)
    p2_policy_configs_sampled = random.sample(
        p2_policy_config_variants, n_mix_per_config
    )
    return p2_policy_configs_sampled


def _assert_all_config_use_the_same_policies(all_rllib_configs, policy_ids):
    for rllib_config in all_rllib_configs:
        assert rllib_config["multiagent"]["policies"].keys() == policy_ids


def _gather_policy_variant_per_policy_id(all_rllib_configs, policy_ids):
    policy_config_variants = {}
    for policy_id in policy_ids:
        policy_config_variants[policy_id] = []
        for rllib_config in all_rllib_configs:
            policy_config_variants[policy_id].append(
                copy.deepcopy(
                    rllib_config["multiagent"]["policies"][policy_id]
                )
            )
    return policy_config_variants
