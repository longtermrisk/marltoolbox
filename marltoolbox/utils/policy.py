from typing import Iterable

import gym
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
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
        def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            config: TrainerConfigDict,
        ):
            print("__init__ FrozenPolicyFromTuneTrainer")

            self.tune_config = config["tune_config"]
            TuneTrainerClass = self.tune_config["TuneTrainerClass"]
            self.tune_trainer = TuneTrainerClass(config=self.tune_config)
            self.load_checkpoint(
                config.pop(LOAD_FROM_CONFIG_KEY, (None, None))
            )
            self._to_log = {}

            super().__init__(observation_space, action_space, config)

        def _compute_action_helper(
            self,
            input_dict,
            *args,
            **kwargs,
        ):
            # print('input_dict["obs"]', input_dict["obs"])
            (
                actions,
                state_out,
                extra_fetches,
            ) = self.tune_trainer.compute_actions(
                self.policy_id, input_dict["obs"]
            )
            return actions, state_out, extra_fetches

        def _initialize_loss_from_dummy_batch(self, *args, **kwargs):
            pass

        def learn_on_batch(self, samples):
            raise NotImplementedError(
                "FrozenPolicyFromTuneTrainer policy can't be trained"
            )

        def get_weights(self):
            return {
                "checkpoint_path": self.checkpoint_path,
                "policy_id": self.policy_id,
            }

        def set_weights(self, weights):
            checkpoint_path = weights["checkpoint_path"]
            policy_id = weights["policy_id"]
            self.load_checkpoint((checkpoint_path, policy_id))

        def load_checkpoint(self, checkpoint_tuple):
            self.checkpoint_path, self.policy_id = checkpoint_tuple
            if self.checkpoint_path is not None:
                self.tune_trainer.load_checkpoint(self.checkpoint_path)

        @property
        def to_log(self):
            to_log = {
                "frozen_policy": self._to_log,
                "nested_tune_policy": {
                    f"policy_{algo_idx}": algo.to_log
                    for algo_idx, algo in enumerate(self.algorithms)
                    if hasattr(algo, "to_log")
                },
            }
            return to_log

        @to_log.setter
        def to_log(self, value):
            if value == {}:
                for algo in self.algorithms:
                    if hasattr(algo, "to_log"):
                        algo.to_log = {}

            self._to_log = value

        def on_episode_start(self, *args, **kwargs):
            if hasattr(self.tune_trainer, "reset_compute_actions_state"):
                self.tune_trainer.reset_compute_actions_state()
            if hasattr(self.tune_trainer, "on_episode_start"):
                self.tune_trainer.on_episode_start()

    return FrozenPolicyFromTuneTrainer


class MyLearningRateSchedule(LearningRateSchedule):
    """
    Mixin for TorchPolicy that adds a learning rate schedule.
    This custom mixin allow to use schedulers other that the linear one
    """

    def __init__(self, lr, lr_schedule):
        self.cur_lr = lr
        if lr_schedule is None:
            self.lr_schedule = ConstantSchedule(lr, framework=None)
        else:
            if isinstance(lr_schedule, Iterable):
                self.lr_schedule = PiecewiseSchedule(
                    lr_schedule,
                    outside_value=lr_schedule[-1][-1],
                    framework=None,
                )
            else:
                self.lr_schedule = lr_schedule


def my_setup_early_mixins(
    policy: Policy, obs_space, action_space, config: TrainerConfigDict
) -> None:
    MyLearningRateSchedule.__init__(
        policy, config["lr"], config["lr_schedule"]
    )
