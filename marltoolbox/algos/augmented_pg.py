from typing import Union, Type, List, Dict

import torch
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.pg import PGTrainer, PGTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.common import _get_global_vars
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils import override
from ray.rllib.utils.metrics import NUM_ENV_STEPS_TRAINED, NUM_ENV_STEPS_SAMPLED
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TrainerConfigDict, TensorType
from ray.util.annotations import DeveloperAPI

from marltoolbox.utils import log
from marltoolbox.utils import optimizers
from marltoolbox.utils.log import get_stats_parameters_from_policy


def my_pg_torch_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """The basic policy gradients loss function.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Pass the training data through our model to get distribution parameters.
    dist_inputs, _ = model(train_batch)

    # Create an action distribution object.
    action_dist = dist_class(dist_inputs, model)
    curr_entropy = action_dist.entropy()

    # Calculate the vanilla PG loss based on:
    # L = -E[ log(pi(a|s)) * A]
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])

    # Final policy loss.
    policy_loss = -torch.mean(log_probs * train_batch[Postprocessing.ADVANTAGES])
    entropy_loss = -policy.entropy_coeff * curr_entropy.mean()

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["policy_loss"] = policy_loss
    model.tower_stats["entropy_loss"] = entropy_loss

    return policy_loss + entropy_loss


def setup_early_mixins(
    policy: Policy, obs_space, action_space, config: TrainerConfigDict
) -> None:
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )


def my_pg_loss_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Returns the calculated loss in a stats dict.

    Args:
        policy (Policy): The Policy object.
        train_batch (SampleBatch): The data used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """

    return {
        "cur_lr": policy.cur_lr,
        "param": get_stats_parameters_from_policy(policy),
        "entropy_coeff": policy.entropy_coeff,
        "policy_loss": torch.mean(torch.stack(policy.get_tower_stats("policy_loss"))),
        "entropy_loss": torch.mean(torch.stack(policy.get_tower_stats("entropy_loss"))),
    }


def fix_add_modules(policy, obs_space, action_space, trainer_config):
    if hasattr(policy.model, "one_hot"):
        for module_idx, module in policy.model.one_hot.items():
            policy.model.add_module("one_hot_{}".format(module_idx), module)
    if hasattr(policy.model, "flatten"):
        for module_idx, module in policy.model.flatten.items():
            policy.model.add_module("flatten_{}".format(module_idx), module)


MyPGTorchPolicyIntermediaryStep = PGTorchPolicy.with_updates(
    optimizer_fn=optimizers.sgd_optimizer_spl,
    before_init=setup_early_mixins,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule],
    stats_fn=log.augment_stats_fn_wt_additionnal_logs(my_pg_loss_stats),
    loss_fn=my_pg_torch_loss,
    after_init=fix_add_modules,
)


class FixSaveLoadWeightMixin:
    @override(Policy)
    @DeveloperAPI
    def set_state(self, state: dict) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            # Fix when using PyTorch > 1.11
            for o, s in zip(self._optimizers, optimizer_vars):
                for v in s["param_groups"]:
                    if "foreach" in v.keys():
                        v["foreach"] = False if v["foreach"] is None else v["foreach"]
                for v in s["state"].values():
                    if "momentum_buffer" in v.keys():
                        v["momentum_buffer"] = (
                            False
                            if v["momentum_buffer"] is None
                            else v["momentum_buffer"]
                        )
                optim_state_dict = convert_to_torch_tensor(s, device=self.device)
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])
        # Then the Policy's (NN) weights.
        super().set_state(state)


class MyPGTorchPolicy(FixSaveLoadWeightMixin, MyPGTorchPolicyIntermediaryStep):
    pass


class MyPGTrainer(PGTrainer):
    _allow_unknown_configs = True

    def training_iteration(self):
        train_results = super().training_iteration()
        global_vars = {"timestep": self._counters[NUM_ENV_STEPS_SAMPLED]}
        self.workers.foreach_worker(lambda w: w.set_global_vars(global_vars))
        return train_results
