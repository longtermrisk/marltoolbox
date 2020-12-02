import pickle
import logging
from typing import Callable

from marltoolbox.utils import miscellaneous
logger = logging.getLogger(__name__)

LOAD_FROM_CONFIG_KEY = "checkpoint_to_load_from"

def _load_checkpoint_from_config(worker):
    for policy_id, policy in worker.policy_map.items():
        checkpoint_path = policy.config.get(LOAD_FROM_CONFIG_KEY, False)
        if checkpoint_path:
            checkpoint = pickle.load(open(checkpoint_path, "rb"))
            assert "worker" in checkpoint.keys()
            assert "optimizer" not in checkpoint.keys()
            objs = checkpoint["worker"]
            objs = pickle.loads(objs)
            # TODO I need to let the user decide to load that too
            # self.sync_filters(objs["filters"])
            found_policy_id = False
            for p_id, state in objs["state"].items():
                if p_id == policy_id:
                    # TODO make logger works
                    logger.warning(f"going to load policy {policy_id} from checkpoint {checkpoint_path}")
                    logger.info(f"going to load policy {policy_id} from checkpoint {checkpoint_path}")
                    policy.set_state(state)
                    found_policy_id = True
            if not found_policy_id:
                logger.warning(f'policy_id {policy_id} not in checkpoint["worker"]["state"].keys() {objs["state"].keys()}')
        else:
            logger.warning(f"no checkpoint found for policy_id: {policy_id} "
                  f"by looking at config key: {LOAD_FROM_CONFIG_KEY}")
def _after_init_load_checkpoint_from_config(trainer):
    trainer.workers.foreach_worker(_load_checkpoint_from_config)


def prepare_trainer_to_load_checkpoints(TrainerClass, existing_after_init_fn: Callable = None):
    """
    :param TrainerClass: the TrainerClass you want to modify to allow custom checkpoint loading
    :param existing_after_init_fn: (optional) the after_init function already used by the provided TrainerClass
    :return: a RLLib TrainerClass which will load the checkpoints
    provided in the policy config under the key defined by LOAD_FROM_CONFIG_KEY
    """
    if existing_after_init_fn is not None:
        # TODO This is not very readable
        TrainerClassWtLoading = TrainerClass.with_updates(
            after_init=(lambda trainer:
                        miscellaneous.sequence_of_fn_wt_same_args(
                            [_after_init_load_checkpoint_from_config, existing_after_init_fn],
                            trainer=trainer)
                        )
        )
    else:
        TrainerClassWtLoading = TrainerClass.with_updates(
            after_init=_after_init_load_checkpoint_from_config)
    return TrainerClassWtLoading