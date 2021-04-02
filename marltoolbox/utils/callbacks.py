from typing import Dict, TYPE_CHECKING

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker


class PolicyCallbacks(DefaultCallbacks):

    def on_episode_start(
            self, worker: "RolloutWorker", *args, **kwargs):
        self._call_method_from_policies(
            worker, "on_episode_start", *args, **kwargs)

    def on_episode_step(
            self, worker: "RolloutWorker", *args, **kwargs):
        self._call_method_from_policies(
            worker, "on_episode_step", *args, **kwargs)

    def on_episode_end(
            self, worker: "RolloutWorker", *args, **kwargs):
        self._call_method_from_policies(
            worker, "on_episode_end", *args, **kwargs)

    def on_postprocess_trajectory(
            self,
            worker: "RolloutWorker",
            agent_id: AgentID,
            policy_id: PolicyID,
            policies: Dict[PolicyID, Policy],
            *args, **kwargs):

        self._call_method_from_policy(
            *args,
            worker=worker,
            method="on_postprocess_trajectory",
            policy=policies[policy_id],
            policy_id=policy_id,
            **kwargs)

    def on_sample_end(
            self, worker: "RolloutWorker", *args, **kwargs):
        self._call_method_from_policies(
            worker, "on_sample_end", *args, **kwargs)

    def on_train_result(self, trainer, *args, **kwargs):
        self._call_method_from_policies(
            *args,
            worker=trainer.workers.local_worker(),
            method="on_train_result",
            trainer=trainer,
            **kwargs)

    def _call_method_from_policies(self, worker, method: str, *args, **kwargs):
        policy_ids = worker.policy_map.keys()
        for policy_id, policy in worker.policy_map.items():
            self._call_method_from_policy(*args,
                                          worker=worker,
                                          method=method,
                                          policy=policy,
                                          policy_id=policy_id,
                                          policy_ids=policy_ids,
                                          **kwargs)

    def _call_method_from_policy(
            self, worker, method: str, policy, policy_id, *args, **kwargs):
        if self._is_callback_implemented_in_policy(policy, method):
            getattr(policy, method)(*args,
                                    worker=worker,
                                    policy=policy,
                                    policy_id=policy_id,
                                    **kwargs)

    def _is_callback_implemented_in_policy(self, policy, callback_method):
        return hasattr(policy, callback_method) and \
               callable(getattr(policy, callback_method))
