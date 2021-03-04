import copy

import random
from ray import tune

from marltoolbox.algos import population
from marltoolbox.utils import miscellaneous


class L1BRConfigurationHelper:

    def __init__(self, rllib_config_to_modif: dict, policy_id_for_lvl0_agents: str, policy_id_for_lvl1_agents: str, ):
        self.rllib_config_to_modif = copy.deepcopy(rllib_config_to_modif)
        self.policy_id_for_lvl0_agents = policy_id_for_lvl0_agents
        self.policy_id_for_lvl1_agents = policy_id_for_lvl1_agents

    def define_exp(self, use_n_lvl0_agents_in_each_population: int, train_n_lvl1_agents: int, lvl0_checkpoints: list,
                   only_use_lvl0_agents_in_one_population: bool = True):

        self.use_n_lvl0_agents_in_each_population = use_n_lvl0_agents_in_each_population
        self.train_n_lvl1_agents = train_n_lvl1_agents
        self.lvl0_checkpoints = lvl0_checkpoints
        self.only_use_lvl0_agents_in_one_population = only_use_lvl0_agents_in_one_population

        assert use_n_lvl0_agents_in_each_population >= 1
        assert train_n_lvl1_agents >= 1
        assert use_n_lvl0_agents_in_each_population <= len(lvl0_checkpoints)
        assert not only_use_lvl0_agents_in_one_population or \
               (use_n_lvl0_agents_in_each_population * train_n_lvl1_agents) <= len(lvl0_checkpoints)

        self.experiment_defined = True

    def prepare_config_for_lvl1_training(self):
        assert self.experiment_defined

        self._no_training_no_exploration_for_lvl0_agents()

        checkpoints_per_population, seed_per_population = self._split_lvl0_checkpoints_into_populations()

        print("WARNING: we overwrite the seeds")
        """
        We need to overwrite the seeds because we use the seeds to select the right group of checkpoints associated 
        with a seed.
        We can't directly write the checkpoints to use in the config since we want to use tune.grid_search over the 
        seeds. Thus we write all the lists of checkpoints in the config (each associated with a population and a seed).
        """
        self._overwrite_seeds(seed_per_population)

        self.rllib_config_to_modif = population.modify_config_to_use_population(
            config=self.rllib_config_to_modif,
            population_policy_id=self.policy_id_for_lvl0_agents,
            opponents_checkpoints=miscellaneous.seed_to_checkpoint(
                checkpoints_per_population)
        )

        return self.rllib_config_to_modif

    def _no_training_no_exploration_for_lvl0_agents(self):
        miscellaneous.overwrite_config(dict_=self.rllib_config_to_modif,
                                       key="multiagent.policies_to_train",
                                       value=[self.policy_id_for_lvl1_agents])
        self.rllib_config_to_modif["multiagent"]["policies"][self.policy_id_for_lvl0_agents][3]["explore"] = False

    def _overwrite_seeds(self, seeds):
        miscellaneous.overwrite_config(dict_=self.rllib_config_to_modif,
                                       key="seed",
                                       value=tune.grid_search(seeds))

    def _split_lvl0_checkpoints_into_populations(self):
        self.checkpoints_idx = self._get_checkpoints_idx()

        if self.only_use_lvl0_agents_in_one_population:
            checkpoints_idx_per_population = self._sample_unique_checkpoints_for_a_population()
        else:
            checkpoints_idx_per_population = self._sample_random_checkpoints_for_each_populations()

        checkpoints_per_population, seed_per_population = self._convert_checkpoints_idx_list_into_checkpoints_list(
            checkpoints_idx_per_population)

        return checkpoints_per_population, seed_per_population

    def _convert_checkpoints_idx_list_into_checkpoints_list(self, checkpoints_idx_per_population):
        checkpoints_per_population = {}
        seed_per_population = []
        for population_n, checkpoints_idx_in_one_population in enumerate(checkpoints_idx_per_population):
            selected_checkpoints = [el for i, el in enumerate(self.lvl0_checkpoints) if i in
                                    checkpoints_idx_in_one_population]
            checkpoints_per_population[population_n] = selected_checkpoints
            seed_per_population.append(population_n)

        assert len(checkpoints_per_population) == self.train_n_lvl1_agents
        assert len(seed_per_population) == self.train_n_lvl1_agents

        return checkpoints_per_population, seed_per_population

    def _get_checkpoints_idx(self):
        checkpoints_idx = list(range(len(self.lvl0_checkpoints)))
        return checkpoints_idx

    def _sample_unique_checkpoints_for_a_population(self):
        checkpoints_idx_remaining = copy.deepcopy(self.checkpoints_idx)
        checkpoints_idx_per_population = []
        for lvl1_i in range(self.train_n_lvl1_agents):
            checkpoints_in_one_population, checkpoints_idx_remaining = \
                self._select_checkpoints_and_remove_from_available(checkpoints_idx_remaining)
            checkpoints_idx_per_population.append(checkpoints_in_one_population)
        return checkpoints_idx_per_population

    def _select_checkpoints_and_remove_from_available(self, checkpoints_idx_remaining):
        checkpoints_in_one_population = random.sample(checkpoints_idx_remaining,
                                                      k=self.use_n_lvl0_agents_in_each_population)
        for checkpoint_idx in checkpoints_in_one_population:
            checkpoints_idx_remaining.remove(checkpoint_idx)
        return checkpoints_in_one_population, checkpoints_idx_remaining

    def _sample_random_checkpoints_for_each_populations(self):
        checkpoints_idx_per_population = []
        for lvl1_i in range(self.train_n_lvl1_agents):
            checkpoints_in_one_population = random.sample(self.checkpoints_idx,
                                                          k=self.use_n_lvl0_agents_in_each_population)
            checkpoints_idx_per_population.append(checkpoints_in_one_population)
        return checkpoints_idx_per_population
