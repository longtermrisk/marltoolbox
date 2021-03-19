import gym
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict

from marltoolbox.utils.restore import LOAD_FROM_CONFIG_KEY


def get_tune_policy_class(PolicyClass):
    """
    Allow to convert a Tune trainer into a frozen RLLib policy
    (no training possible).

    :param PolicyClass: The base RLLib policy class to use.
    Can be needed if you need some statistics or postprocessing.
    :return: an RLLib policy class that compute actions by calling the
    Tune trainer.
    """

    class FrozenPolicyFromTuneTrainer(PolicyClass):

        def __init__(self, observation_space: gym.spaces.Space,
                     action_space: gym.spaces.Space,
                     config: TrainerConfigDict):
            print("__init__ FrozenPolicyFromTuneTrainer")

            self.tune_config = config["tune_config"]
            TuneTrainerClass = self.tune_config["TuneTrainerClass"]
            self.tune_trainer = TuneTrainerClass(config=self.tune_config)
            self.load_checkpoint(
                config.pop(LOAD_FROM_CONFIG_KEY, (None, None)))

            super().__init__(observation_space, action_space, config)

        def compute_actions(self,
                            obs_batch,
                            state_batches=None,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):
            actions, state_out, extra_fetches = \
                self.tune_trainer.compute_actions(self.policy_id, obs_batch)
            return actions, state_out, extra_fetches

        def learn_on_batch(self, samples):
            raise NotImplementedError(
                "FrozenPolicyFromTuneTrainer policy can't be trained")

        def get_weights(self):
            return {"checkpoint_path": self.checkpoint_path,
                    "policy_id": self.policy_id}

        def set_weights(self, weights):
            checkpoint_path = weights["checkpoint_path"]
            policy_id = weights["policy_id"]
            self.load_checkpoint((checkpoint_path, policy_id))

        def load_checkpoint(self, checkpoint_tuple):
            self.checkpoint_path, self.policy_id = checkpoint_tuple
            if self.checkpoint_path is not None:
                self.tune_trainer.load_checkpoint(self.checkpoint_path)

    return FrozenPolicyFromTuneTrainer


import gym
from typing import Iterable

from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
from ray.rllib.utils.typing import TrainerConfigDict

torch, _ = try_import_torch()


class MyLearningRateSchedule(LearningRateSchedule):
    """Mixin for TFPolicy that adds a learning rate schedule."""
    def __init__(self, lr, lr_schedule):
        self.cur_lr = lr
        if lr_schedule is None:
            self.lr_schedule = ConstantSchedule(lr, framework=None)
        else:
            if isinstance(lr_schedule, Iterable):
                self.lr_schedule = PiecewiseSchedule(
                    lr_schedule, outside_value=lr_schedule[-1][-1],
                    framework=None)
            else:
                self.lr_schedule = lr_schedule

def my_setup_early_mixins(policy: Policy, obs_space, action_space,
                       config: TrainerConfigDict) -> None:
    MyLearningRateSchedule.__init__(policy,
                                    config["lr"],
                                    config["lr_schedule"])