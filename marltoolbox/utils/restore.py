import pickle
import logging

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
            for pid, state in objs["state"].items():
                if pid == policy_id:
                    # TODO make logger works
                    print(f"going to load policy {policy_id} from checkpoint {checkpoint_path}")
                    logger.info(f"going to load policy {policy_id} from checkpoint {checkpoint_path}")
                    policy.set_state(state)
            if not found_policy_id:
                logger.warning(f'policy_id {policy_id} not in checkpoint["worker"]["state"].keys() {objs["state"].keys()}')
        else:
            print("no checkpoint found for policy_id:", policy_id,
                  "by looking at config key:", LOAD_FROM_CONFIG_KEY)
def after_init_load_checkpoint_from_config(trainer):
    trainer.workers.foreach_worker(_load_checkpoint_from_config)
