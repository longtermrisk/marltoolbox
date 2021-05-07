import logging

import torch
from ray.rllib.policy.torch_policy import TorchPolicy
from torch.nn import Module
from ray.rllib.evaluation import RolloutWorker

logger = logging.getLogger(__name__)


class ModelSummarizer:
    """
    Helper to log for every torch.nn modules in every policies the
    architecture and some parameter statistics.
    """

    @staticmethod
    def for_every_policy_print_model_stats(worker: RolloutWorker):
        """
        For every policies in the worker, log the archi of all torch modules
        and some statistiques about their parameters

        :param worker:
        """
        for policy_id, policy in worker.policy_map.items():
            msg = f"===== Models summaries policy_id {policy_id} ====="
            print(msg)
            logger.info(msg)
            ModelSummarizer._print_model_summary(policy)
            ModelSummarizer._count_parameters_in_every_modules(policy)

    @staticmethod
    def _print_model_summary(policy: TorchPolicy):
        if isinstance(policy, TorchPolicy):
            for k, v in policy.__dict__.items():
                if isinstance(v, Module):
                    msg = f"{k}, {v}"
                    print(msg)
                    logger.info(msg)

    @staticmethod
    def _count_parameters_in_every_modules(policy: TorchPolicy):
        if isinstance(policy, TorchPolicy):
            for k, v in policy.__dict__.items():
                if isinstance(v, Module):
                    ModelSummarizer._count_and_log_for_one_module(policy, k, v)

    @staticmethod
    def _count_and_log_for_one_module(
        policy: TorchPolicy, module_name: str, module: torch.nn.Module
    ):
        n_param = ModelSummarizer._count_parameters(module, module_name)
        n_param_shared_counted_once = ModelSummarizer._count_parameters(
            module, module_name, count_shared_once=True
        )
        n_param_trainable = ModelSummarizer._count_parameters(
            module, module_name, only_trainable=True
        )
        ModelSummarizer._log_values_in_to_log(
            policy,
            {
                f"module_{module_name}_n_param": n_param,
                f"module_{module_name}_n_param_shared_counted_once": n_param_shared_counted_once,
                f"module_{module_name}_n_param_trainable": n_param_trainable,
            },
        )

    @staticmethod
    def _log_values_in_to_log(policy, dictionary):
        if hasattr(policy, "to_log"):
            policy.to_log.update(dictionary)

    @staticmethod
    def _count_parameters(
        m: torch.nn.Module,
        module_name: str,
        count_shared_once: bool = False,
        only_trainable: bool = False,
    ):
        """
        returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = m.parameters()
        if only_trainable:
            parameters = list(p for p in parameters if p.requires_grad)
        if count_shared_once:
            parameters = dict((p.data_ptr(), p) for p in parameters).values()
        number_of_parameters = sum(p.numel() for p in parameters)

        msg = (
            f"{module_name}: "
            f"number_of_parameters: {number_of_parameters} "
            f"(only_trainable: {only_trainable}, "
            f"count_shared_once: {count_shared_once})"
        )
        print(msg)
        logger.info(msg)
        return number_of_parameters
