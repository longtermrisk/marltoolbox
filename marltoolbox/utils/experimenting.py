import copy
import random
from typing import Callable, Optional

import ray
from ray.rllib.agents.pg import PGTrainer

from marltoolbox.utils.restore import after_init_load_checkpoint_from_config, LOAD_FROM_CONFIG_KEY


def sequence_of_fn_wt_same_args(function_list, *args, **kwargs) -> None:
    for fn in function_list:
        fn(*args, **kwargs)


def set_config_for_evaluation(config: dict) -> dict:
    config_copy = copy.deepcopy(config)

    # Do not train
    # Always multiagent
    assert "multiagent" in config_copy.keys(), "Only working for config with multiagent key. " \
                                               f"config_copy.keys(): {config_copy.keys()}"
    config_copy["multiagent"]["policies_to_train"] = []

    # Setup for evaluation
    # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    config_copy["explore"] = False

    # TODO below is really useless? If so then clean it
    # The following is not really needed since we are not training any policies
    # === Optimization ===
    # Learning rate for adam optimizer
    # config_copy["lr"] = 0.0
    # # Learning rate schedule
    # if "lr_schedule" in trainer_default_config.keys():
    #     config_copy["lr_schedule"] = None

    return config_copy


def prepare_trainer_to_load_checkpoints(TrainerClass, additionnal_after_init_fn):
    if additionnal_after_init_fn is not None:
        # TODO This is not very readable
        TrainerClassWtLoading = TrainerClass.with_updates(
            after_init=(lambda trainer:
                        sequence_of_fn_wt_same_args(
                            [after_init_load_checkpoint_from_config, additionnal_after_init_fn],
                            trainer=trainer)
                        )
        )
    else:
        TrainerClassWtLoading = TrainerClass.with_updates(
            after_init=after_init_load_checkpoint_from_config)
    return TrainerClassWtLoading


class SameAndCrossPlay:
    """
    Does support using RLLib policies only or RLLib Policies
    """

    def __init__(self, evaluation_config: dict, stop_config: dict,
                 exp_name: str,
                 TuneTrainerClass = None,
                 TrainerClass = PGTrainer,
                 extract_checkpoints_from_results=None,
                 checkpoint_list: list = None,
                 additionnal_after_init_fn: Optional[Callable] = None,
                 n_same_play_per_checkpoint=1, n_cross_play_per_checkpoint=1,
                 call_after_init=True):
        """
        Works for a unique pair of RLLib policies
        # TODO parallelize it with Ray/Tune?
        # TODO docstring
        """
        # Either extract_checkpoints_from_results or checkpoint_list
        assert extract_checkpoints_from_results or checkpoint_list
        if extract_checkpoints_from_results:
            assert checkpoint_list is None
            checkpoint_list = self.extract_checkpoints(extract_checkpoints_from_results)
        assert n_same_play_per_checkpoint + n_cross_play_per_checkpoint >= 1
        assert n_same_play_per_checkpoint >= 0
        assert n_cross_play_per_checkpoint >= 0
        if n_cross_play_per_checkpoint > 0:
            assert len(checkpoint_list) > 1
        assert "multiagent" in evaluation_config.keys()
        assert len(evaluation_config["multiagent"]["policies"].keys()) == 2

        self.TuneTrainerClass = TuneTrainerClass
        TrainerClassWtCheckpointLoading = prepare_trainer_to_load_checkpoints(TrainerClass, additionnal_after_init_fn)
        self.TrainerClass = TrainerClassWtCheckpointLoading

        evaluation_config = set_config_for_evaluation(evaluation_config)

        self.evaluation_config = evaluation_config
        self.stop_config = stop_config
        self.exp_name = exp_name
        self.checkpoint_list = checkpoint_list
        self.n_same_play_per_checkpoint = n_same_play_per_checkpoint
        self.n_cross_play_per_checkpoint = n_cross_play_per_checkpoint

        if call_after_init:
            self.__call__()

    def extract_checkpoints(self, results):
        all_best_checkpoints_per_trial = [results.get_best_checkpoint(trial,
                                                            metric=results.default_metric,
                                                            mode=results.default_mode)
                                for trial in results.trials]
        return all_best_checkpoints_per_trial

    def __call__(self, n_same_play_per_checkpoint=None, n_cross_play_per_checkpoint=None, checkpoint_list=None):
        """
        Works for a unique pair of policies
        # TODO docstring
        """
        if n_same_play_per_checkpoint is None:
            n_same_play_per_checkpoint = self.n_same_play_per_checkpoint
        if n_cross_play_per_checkpoint is None:
            n_cross_play_per_checkpoint = self.n_cross_play_per_checkpoint
        if checkpoint_list is None:
            checkpoint_list = self.checkpoint_list

        all_results = []
        for checkpoint_i, checkpoint_path in enumerate(checkpoint_list):

            for same_play_n in range(n_same_play_per_checkpoint):
                results_wt_metadata = self._run_one_same_play(checkpoint_path, checkpoint_i)
                all_results.append(results_wt_metadata)

            for cross_play_n in range(n_cross_play_per_checkpoint):
                results_wt_metadata = self._run_one_cross_play(checkpoint_path, checkpoint_i)
                all_results.append(results_wt_metadata)

        print("all_results", all_results)
        # TODO add analysis and plotting

    def _run_one_same_play(self, checkpoint_path, checkpoint_i) -> dict:
        results_wt_metadata = {"mode": "same-play"}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Add the checkpoints to load from in the policy_config
        for policy_id in config_copy["multiagent"]["policies"].keys():
            config_copy["multiagent"]["policies"][policy_id][3][LOAD_FROM_CONFIG_KEY] = checkpoint_path
            config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
            config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass
            results_wt_metadata[policy_id] = {"checkpoint_path": checkpoint_path, "checkpoint_i": checkpoint_i}

        results = ray.tune.run(self.TrainerClass, config=config_copy,
                               stop=self.stop_config, verbose=1, name=f"{self.exp_name}_same_play")

        results_wt_metadata["results"] = results
        return results_wt_metadata


    def _run_one_cross_play(self, checkpoint_path, checkpoint_i) -> dict:
        results_wt_metadata = {"mode": "cross-play"}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Select opponent
        opponent_i = checkpoint_i
        while opponent_i == checkpoint_i:
            opponent_i = random.randint(0, len(self.checkpoint_list) - 1)

        selected_selection_order = random.randint(0, len(config_copy["multiagent"]["policies"]) - 1)

        # Add the checkpoints to load from in the policy_config
        policy_order = 0
        for policy_id in config_copy["multiagent"]["policies"].keys():

            if policy_order == selected_selection_order:
                selected_checkpoint_path = checkpoint_path
                results_wt_metadata[policy_id] = {"checkpoint_path": selected_checkpoint_path,
                                                  "checkpoint_i": checkpoint_i}
            else:
                selected_checkpoint_path = self.checkpoint_list[opponent_i]
                results_wt_metadata[policy_id] = {"checkpoint_path": selected_checkpoint_path,
                                                  "checkpoint_i": opponent_i}

            config_copy["multiagent"]["policies"][policy_id][3][LOAD_FROM_CONFIG_KEY] = selected_checkpoint_path
            config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
            config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass

        results = ray.tune.run(self.TrainerClass, config=config_copy,
                               stop=self.stop_config, verbose=1, name=f"{self.exp_name}_cross_play")

        results_wt_metadata["results"] = results
        return results_wt_metadata


