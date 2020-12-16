import copy
import copy

import random
from ray import tune

from marltoolbox.algos import population
from marltoolbox.utils import miscellaneous

# TODO is that useful?
def config_lvl0_population_training(n_seeds: int, lvl0_kwargs: dict):
    print("Training lvl0_population")
    seeds = list(range(n_seeds))
    miscellaneous.overwrite_config(dict_=lvl0_kwargs,
                                   key="config.seed", value=tune.grid_search(seeds))
    miscellaneous.overwrite_config(dict_=lvl0_kwargs,
                                   key="checkpoint_freq", value=0)
    miscellaneous.overwrite_config(dict_=lvl0_kwargs,
                                   key="checkpoint_at_end", value=True)
    miscellaneous.overwrite_config(dict_=lvl0_kwargs,
                                   key="metric", value="episode_reward_mean")
    miscellaneous.overwrite_config(dict_=lvl0_kwargs,
                                   key="mode", value="max")


def prepare_config_for_lvl1_training(config: dict, lvl0_policy_id: str, lvl1_policy_id: str,
                                     select_n_lvl0_from_population: int, n_lvl1_to_train: int,
                                     overlapping_population: bool, lvl0_checkpoints: list,
                                     overwrite_seeds=False):
    assert select_n_lvl0_from_population >= 1 and isinstance(select_n_lvl0_from_population, int)
    assert n_lvl1_to_train >= 1 and isinstance(n_lvl1_to_train, int)
    assert select_n_lvl0_from_population <= len(lvl0_checkpoints)
    assert overlapping_population or (select_n_lvl0_from_population * n_lvl1_to_train) <= len(lvl0_checkpoints)

    miscellaneous.overwrite_config(dict_=config,
                                   key="multiagent.policies_to_train",
                                   value=[lvl1_policy_id])
    config["multiagent"]["policies"][lvl0_policy_id][3]["explore"] = False

    miscellaneous.overwrite_config(dict_=config,
                                   key="seed",
                                   value=tune.grid_search(list(range(n_lvl1_to_train))))

    checkpoints_idx_per_lvl1 = _checkpoints_splitter(
        lvl0_checkpoints=lvl0_checkpoints,
        n_lvl0_in_population=select_n_lvl0_from_population,
        n_lvl1_to_train=n_lvl1_to_train,
        overlapping_population=overlapping_population)

    checkpoints_list = []
    for idx_list in checkpoints_idx_per_lvl1:
        selected_checkpoints = [el for i, el in enumerate(lvl0_checkpoints) if i in idx_list]
        checkpoints_list.append(selected_checkpoints)

    population.replace_opponent_by_population_of_opponents(
        config=config,
        opponent_policy_id=lvl0_policy_id,
        opponents_checkpoints=miscellaneous.use_seed_as_idx(checkpoints_list))

    return config


def _checkpoints_splitter(lvl0_checkpoints: int, n_lvl0_in_population: int,
                          n_lvl1_to_train: int, overlapping_population: bool):
    checkpoints_idx = list(range(len(lvl0_checkpoints)))
    checkpoints_idx_available = checkpoints_idx

    checkpoints_idx_per_lvl1 = []
    for lvl1_i in range(n_lvl1_to_train):
        if overlapping_population:
            opponents_checkpoints = random.sample(checkpoints_idx, k=n_lvl0_in_population)
        else:
            opponents_checkpoints = random.sample(checkpoints_idx_available, k=n_lvl0_in_population)
            for checkpoint_value in opponents_checkpoints:
                checkpoints_idx_available.remove(checkpoint_value)
        checkpoints_idx_per_lvl1.append(opponents_checkpoints)
    return checkpoints_idx_per_lvl1
