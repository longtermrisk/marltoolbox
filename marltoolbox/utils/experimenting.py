import copy
import logging
import random
from typing import Callable, Optional, Tuple

from collections import Iterable

# TODO add matplotlib in setup in full install version
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import ray
from ray.rllib.agents.pg import PGTrainer


from marltoolbox.utils import restore, log

logger = logging.getLogger(__name__)




def set_config_for_evaluation(config: dict, policies_to_train=["None"]) -> dict:
    config_copy = copy.deepcopy(config)

    # Do not train
    # Always multiagent
    assert "multiagent" in config_copy.keys(), "Only working for config with multiagent key. " \
                                               f"config_copy.keys(): {config_copy.keys()}"
    config_copy["multiagent"]["policies_to_train"] = policies_to_train

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
    config_copy["lr"] = 0.0
    # # Learning rate schedule
    if "lr_schedule" in config_copy.keys():
        config_copy["lr_schedule"] = None

    return config_copy




class SameAndCrossPlay:
    """
    Does support using RLLib policies only or RLLib Policies

    """
    # TODO add function to list all available metrics
    SELF_PLAY_MODE = "same-play"
    CROSS_PLAY_MODE = "cross-play"
    MODES = [SELF_PLAY_MODE, CROSS_PLAY_MODE]

    def __init__(self, evaluation_config: dict, stop_config: dict, exp_name: str,
                 metrics: Tuple[Tuple[str]], metric_mode: str="avg",
                 colors=["blue", "red"],
                 TuneTrainerClass=None, TrainerClass=None,
                 extract_checkpoints_from_results=None,
                 checkpoint_list: list = None,
                 additionnal_after_init_fn: Optional[Callable] = None,
                 n_same_play_per_checkpoint: int = 1,
                 n_cross_play_per_checkpoint: int = 1,
                 call_after_init: bool = True,
                 mix_policy_order: bool = False,
                 policies_to_train: list = ["None"],
                 policies_to_load: list = ["All"]):
        """
        Works for a unique pair of RLLib policies
        # TODO parallelize it with Ray/Tune?
        # TODO docstring
        :param evaluation_config:
        :param stop_config: strop config to give to Tune.run(stop=stop_config,...)
        :param exp_name:
        :param metrics: Tuple of pair of metrics to plot such as: ((x_metric, y_metric),...)
        :param metric_mode: one of ("max", "min", "avg", "last", "last-5-avg", "last-10-avg")
        :param TuneTrainerClass:
        :param TrainerClass:
        :param extract_checkpoints_from_results:
        :param checkpoint_list:
        :param additionnal_after_init_fn:
        :param n_same_play_per_checkpoint:
        :param n_cross_play_per_checkpoint:
        :param call_after_init:
        :param mix_policy_order:
        :param policies_to_train:
        :param policies_to_load:
        """

        self.mix_policy_order = mix_policy_order
        self.policies_to_train = policies_to_train
        self.policies_to_load = policies_to_load

        # Either extract_checkpoints_from_results or checkpoint_list
        assert extract_checkpoints_from_results or checkpoint_list
        if extract_checkpoints_from_results:
            assert checkpoint_list is None
            checkpoint_list = self.extract_checkpoints(extract_checkpoints_from_results)
        self.checkpoint_list = checkpoint_list

        if TrainerClass is None:
            TrainerClass = PGTrainer
        TrainerClassWtCheckpointLoading = restore.prepare_trainer_to_load_checkpoints(TrainerClass,
                                                                                 additionnal_after_init_fn)
        self.TrainerClass = TrainerClassWtCheckpointLoading
        self.TuneTrainerClass = TuneTrainerClass

        assert "multiagent" in evaluation_config.keys()
        assert len(evaluation_config["multiagent"]["policies"].keys()) == 2
        evaluation_config = set_config_for_evaluation(evaluation_config, policies_to_train)
        self.evaluation_config = evaluation_config
        self.stop_config = stop_config
        self.exp_name = exp_name

        assert n_same_play_per_checkpoint + n_cross_play_per_checkpoint >= 1
        assert n_same_play_per_checkpoint >= 0
        assert n_cross_play_per_checkpoint >= 0
        if n_cross_play_per_checkpoint > 0:
            assert len(checkpoint_list) >= n_cross_play_per_checkpoint
        self.n_same_play_per_checkpoint = n_same_play_per_checkpoint
        self.n_cross_play_per_checkpoint = n_cross_play_per_checkpoint

        assert isinstance(metrics, Iterable)
        assert len(metrics) > 0
        for metrics_pair in metrics:
            assert len(metrics_pair) == 2
        self.metrics = metrics
        self.metric_mode = metric_mode
        self.colors = colors

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

        self.plot_results(all_results)
        # TODO add analysis and plotting

    def _run_one_same_play(self, checkpoint_path, checkpoint_i) -> dict:
        results_wt_metadata = {"mode": self.SELF_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Add the checkpoints to load from in the policy_config
        for policy_id in config_copy["multiagent"]["policies"].keys():
            if policy_id in self.policies_to_load or "All" in self.policies_to_load:
                results_wt_metadata[policy_id] = {"checkpoint_path": checkpoint_path, "checkpoint_i": checkpoint_i}
                config_copy["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = checkpoint_path
                config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
                config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass

        results = ray.tune.run(self.TrainerClass, config=config_copy,
                               stop=self.stop_config, verbose=1, name=f"{self.exp_name}_same_play")

        results_wt_metadata["results"] = results
        return results_wt_metadata

    def _run_one_cross_play(self, checkpoint_path, checkpoint_i) -> dict:
        results_wt_metadata = {"mode": self.CROSS_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Select opponent
        opponent_i = checkpoint_i
        while opponent_i == checkpoint_i:
            opponent_i = random.randint(0, len(self.checkpoint_list) - 1)

        if self.mix_policy_order:
            selected_selection_order = random.randint(0, len(config_copy["multiagent"]["policies"]) - 1)
        else:
            selected_selection_order = 0
        # Add the checkpoints to load from in the policy_config
        policy_order = 0
        for policy_id in config_copy["multiagent"]["policies"].keys():
            if policy_id in self.policies_to_load or "All" in self.policies_to_load:
                if policy_order == selected_selection_order:
                    selected_checkpoint_path = checkpoint_path
                    results_wt_metadata[policy_id] = {"checkpoint_path": selected_checkpoint_path,
                                                      "checkpoint_i": checkpoint_i}
                else:
                    selected_checkpoint_path = self.checkpoint_list[opponent_i]
                    results_wt_metadata[policy_id] = {"checkpoint_path": selected_checkpoint_path,
                                                      "checkpoint_i": opponent_i}
                config_copy["multiagent"]["policies"][policy_id][3][restore.LOAD_FROM_CONFIG_KEY] = \
                    selected_checkpoint_path
                config_copy["multiagent"]["policies"][policy_id][3]["policy_id"] = policy_id
                config_copy["multiagent"]["policies"][policy_id][3]["TuneTrainerClass"] = self.TuneTrainerClass

        results = ray.tune.run(self.TrainerClass, config=config_copy,
                               stop=self.stop_config, verbose=1, name=f"{self.exp_name}_cross_play")

        results_wt_metadata["results"] = results
        return results_wt_metadata

    # TODO may be make the plotting independant to be used for other stuff as well
    def plot_results(self, all_results: dict):

        for metric_x, metric_y in self.metrics:
            analysis_per_mode = []
            for mode in self.MODES:
                analysis_list = [report["results"] for report in all_results if report["mode"] == mode]
                analysis_per_mode.append((mode, analysis_list))

            for mode_i, mode_data in enumerate(analysis_per_mode):
                mode, analysis_list = mode_data
                x, y = [], []
                if len(analysis_list) > 0:
                    for analysis in analysis_list:

                        trials = analysis.trials
                        assert len(trials) == 1
                        trial = trials[0]
                        print(trial.metric_analysis)

                        x.append(trial.metric_analysis[metric_x][self.metric_mode])
                        y.append(trial.metric_analysis[metric_y][self.metric_mode])

                plt.plot(x, y, 'o', color=self.colors[mode_i], label=mode, alpha=0.33)
            plt.legend(numpoints=1)
            plt.title("Same-play vs Cross-play")
            plt.xlabel(metric_x)
            plt.ylabel(metric_y)
            plt.show()

