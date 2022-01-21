import logging
import os
import pickle
from typing import List

from marltoolbox import utils
from marltoolbox.utils import path
from ray.tune.analysis import ExperimentAnalysis

logger = logging.getLogger(__name__)

LOAD_FROM_CONFIG_KEY = "checkpoint_to_load_from"


def before_loss_init_load_policy_checkpoint(
    policy, observation_space=None, action_space=None, trainer_config=None
):
    """
    This function is to be given to a policy template(a policy factory)
    (to the 'after_init' argument).
    It will load a specific policy state from a given checkpoint
    (instead of all policies like what does the restore option provided by
    RLLib).

    The policy config must contain the tuple (checkpoint_path, policy_id) to
    load from, stored under the LOAD_FROM_CONFIG_KEY key.

    Finally, the checkpoint_path can be callable, in this case it must
    return a path (str) and accept the policy config as the only argument.
    This last feature allows to dynamically select checkpoints
    for example in multistage training or experiments
    Example: determining the checkpoint to load conditional on the current seed
    (when doing a grid_search over random seeds and with a multistage training)
    """
    checkpoint_path, policy_id = policy.config.get(
        LOAD_FROM_CONFIG_KEY, (None, None)
    )

    if callable(checkpoint_path):
        checkpoint_path = checkpoint_path(policy.config)

    if checkpoint_path is not None:
        load_one_policy_checkpoint(policy_id, policy, checkpoint_path)
        msg = (
            f"marltoolbox restore: checkpoint found for policy_id: "
            f"{policy_id}"
        )
        logger.debug(msg)
    else:
        msg = (
            f"marltoolbox restore: NO checkpoint found for policy_id:"
            f" {policy_id} and policy {policy}."
            f"Not found under the config key: {LOAD_FROM_CONFIG_KEY}"
        )
        logger.warning(msg)


def load_one_policy_checkpoint(
    policy_id, policy, checkpoint_path, using_Tune_class=False
):
    """

    :param policy_id: the policy_id of the policy inside the checkpoint that
        is going to be loaded into the policy provided as 2nd argument
    :param policy: the policy to load the checkpoint into
    :param checkpoint_path: the checkpoint to load from
    :param using_Tune_class: to be set to True in case you are loading a
        policy from a Tune checkpoint
        (not a RLLib checkpoint) and that the policy you are loading into was
        created by converting your Tune trainer
        into frozen a RLLib policy
    :return: None
    """
    if using_Tune_class:
        # The provided policy must implement load_checkpoint.
        # This is only intended for the policy class:
        # FrozenPolicyFromTuneTrainer
        policy.load_checkpoint(checkpoint_tuple=(checkpoint_path, policy_id))
    else:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        logger.debug(f"checkpoint_path {checkpoint_path}")
        checkpoint = pickle.load(open(checkpoint_path, "rb"))
        assert "worker" in checkpoint.keys()
        assert "optimizer" not in checkpoint.keys()
        objs = pickle.loads(checkpoint["worker"])
        # TODO Should let the user decide to load that too
        # self.sync_filters(objs["filters"])
        logger.warning("restoring ckpt: not loading objs['filters']")
        found_policy_id = False
        for p_id, state in objs["state"].items():
            if p_id == policy_id:
                logger.debug(
                    f"going to load policy {policy_id} "
                    f"from checkpoint {checkpoint_path}"
                )
                policy.set_state(state)
                found_policy_id = True
                break
        if not found_policy_id:
            logger.debug(
                f"policy_id {policy_id} not in "
                f'checkpoint["worker"]["state"].keys() '
                f'{objs["state"].keys()}'
            )


def extract_checkpoints_from_experiment_analysis(
    tune_experiment_analysis: ExperimentAnalysis,
) -> List[str]:
    """
    Extract all the best checkpoints from a tune analysis object. This tune
    analysis can contains several trials. Each trial can contains several
    checkpoitn, only the best checkpoint per trial is returned.

    :param tune_experiment_analysis:
    :return: list of all the unique best checkpoints for each trials in the
        tune analysis.
    """
    logger.info("start extract_checkpoints")

    for trial in tune_experiment_analysis.trials:
        checkpoints = tune_experiment_analysis.get_trial_checkpoints_paths(
            trial, tune_experiment_analysis.default_metric
        )
        assert len(checkpoints) > 0

    all_best_checkpoints_per_trial = [
        tune_experiment_analysis.get_best_checkpoint(
            trial,
            metric=tune_experiment_analysis.default_metric,
            mode=tune_experiment_analysis.default_mode,
        )
        for trial in tune_experiment_analysis.trials
    ]

    for checkpoint in all_best_checkpoints_per_trial:
        assert checkpoint is not None

    logger.info("end extract_checkpoints")
    return all_best_checkpoints_per_trial


def get_checkpoint_for_each_replicates(
    all_replicates_save_dir: List[str],
) -> List[str]:
    """
    Get the list of paths to the checkpoint files inside an experiment dir of
    RLLib/Tune (which can contains several trials).
    Works for an experiment with trials containing an unique checkpoint.

    :param all_replicates_save_dir: trial dir
    :return: list of paths to checkpoint files
    """
    ckpt_dir_per_replicate = []
    for replicate_dir_path in all_replicates_save_dir:
        ckpt_dir_path = get_ckpt_dir_for_one_replicate(replicate_dir_path)
        ckpt_path = get_ckpt_from_ckpt_dir(ckpt_dir_path)
        ckpt_dir_per_replicate.append(ckpt_path)
    return ckpt_dir_per_replicate


def get_ckpt_dir_for_one_replicate(replicate_dir_path: str) -> str:
    """
    Get the path to the unique checkpoint dir inside a trial dir of RLLib/Tune.

    :param replicate_dir_path: trial dir
    :return: path to checkpoint dir
    """
    partialy_filtered_ckpt_dir = (
        utils.path.get_children_paths_wt_selecting_filter(
            replicate_dir_path, _filter="checkpoint_"
        )
    )
    ckpt_dir = [
        file_path
        for file_path in partialy_filtered_ckpt_dir
        if ".is_checkpoint" not in file_path
    ]
    assert len(ckpt_dir) == 1, f"{ckpt_dir}"
    return ckpt_dir[0]


def get_ckpt_from_ckpt_dir(ckpt_dir_path: str) -> str:
    """
    Get the path to the unique checkpoint file inside a checkpoint dir of
    RLLib/Tune
    :param ckpt_dir_path: checkpoint dir
    :return: path to checkpoint file
    """
    partialy_filtered_ckpt_path = (
        utils.path.get_children_paths_wt_discarding_filter(
            ckpt_dir_path, _filter="tune_metadata"
        )
    )
    filters = [
        # For Tune/RLLib
        ".is_checkpoint",
        # For TensorFlow
        "ckpt.index",
        "ckpt.data-",
        "ckpt.meta",
        ".json",
    ]
    ckpt_path = filter(
        lambda el: all(filter_ not in el for filter_ in filters),
        partialy_filtered_ckpt_path,
    )
    ckpt_path = list(ckpt_path)
    assert len(ckpt_path) == 1, f"{ckpt_path}"
    return ckpt_path[0]
