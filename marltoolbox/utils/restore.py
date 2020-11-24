import pickle
import logging

logger = logging.getLogger(__name__)


#
# def after_init_load_weights_and_set_lr(trainer: Trainer) -> None:
#     """
#     To use after restoring a model (useful it has a LRRateScheduler)
#     """
#     if trainer.workers.remote_workers():
#         weights = ray.put(trainer.workers.local_worker().get_weights(
#             trainer.policies))
#         for e in trainer.workers.remote_workers():
#             e.set_weights.remote(weights, _get_global_vars())
#
#     trainer.workers.local_worker().set_global_vars(_get_global_vars())

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
            for pid, state in objs["state"].items():
                if pid == policy_id:
                    # TODO make logger works
                    print(f"going to load policy {policy_id} from checkpoint {checkpoint_path}")
                    logger.info(f"going to load policy {policy_id} from checkpoint {checkpoint_path}")
                    policy.set_state(state)
            if not found_policy_id:
                logger.warning(f'policy_id {policy_id} not in checkpoint["worker"]["state"].keys() {objs["state"].keys()}')

def after_init_load_checkpoint_from_config(trainer):
    trainer.workers.foreach_worker(_load_checkpoint_from_config)
