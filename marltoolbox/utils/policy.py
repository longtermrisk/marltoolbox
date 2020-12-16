
import gym

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy

from ray.rllib.utils.typing import AgentID, ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict, Tuple, Union

from marltoolbox.utils.restore import  LOAD_FROM_CONFIG_KEY

# TODO refactor to have a module with all the stuff to suport Tune classes

# class FreezedPolicyFromTuneTrainer(TorchPolicy):
#
#     def __init__(self, observation_space: gym.spaces.Space,
#                  action_space: gym.spaces.Space, config: TrainerConfigDict):
#         print("__init__ FreezedPolicyFromTuneTrainer")
#
#         TuneTrainerClass = config["TuneTrainerClass"]
#         tune_trainer = TuneTrainerClass(config=config)
#         # remove checkpoint from config to prevent after_init_load_checkpoint_from_config
#         # to try to load a RLLib checkpoint
#         # TODO make this cleaner, there should be no need to know about after_init_load_checkpoint_from_config,
#         #  remove side effects
#
#         self.tune_trainer = tune_trainer
#         self.load_checkpoint(config.pop(LOAD_FROM_CONFIG_KEY, (None, None)))
#
#         super().__init__(observation_space, action_space, config)
#
#     def compute_actions(self,
#                         obs_batch,
#                         state_batches=None,
#                         prev_action_batch=None,
#                         prev_reward_batch=None,
#                         info_batch=None,
#                         episodes=None,
#                         **kwargs):
#         actions, state_out, extra_fetches = self.tune_trainer.compute_actions(self.policy_id, obs_batch)
#         return actions, state_out, extra_fetches
#
#     def learn_on_batch(self, samples):
#         raise NotImplementedError("FreezedPolicyFromTuneTrainer policy can't be trained")
#
#     def get_weights(self):
#         raise NotImplementedError("Not supported. Using tune_trainer.load_checkpoint in __init__ to load weights")
#
#     def set_weights(self, weights):
#         raise NotImplementedError("Not supported. Using tune_trainer.load_checkpoint in __init__ to load weights")
#
#     def load_checkpoint(self, checkpoint_tuple):
#         self.checkpoint_path, self.policy_id = checkpoint_tuple
#         if self.checkpoint_path is not None:
#             self.tune_trainer.load_checkpoint(self.checkpoint_path)
#
#     # @property
#     # def model(self):
#     #     return self.tune_trainer.model

#TODO add something to not load and create everything when only evaluating with RLLib
def get_tune_policy_class(PolicyClass):

    class FreezedPolicyFromTuneTrainer(PolicyClass):

        def __init__(self, observation_space: gym.spaces.Space,
                     action_space: gym.spaces.Space, config: TrainerConfigDict):
            print("__init__ FreezedPolicyFromTuneTrainer")

            self.tune_config = config["tune_config"]
            TuneTrainerClass = self.tune_config["TuneTrainerClass"]
            self.tune_trainer = TuneTrainerClass(config=self.tune_config)

            # remove checkpoint from config to prevent after_init_load_checkpoint_from_config
            # to try to load a RLLib checkpoint
            # TODO make this cleaner, there should be no need to know about after_init_load_checkpoint_from_config,
            #  remove side effects

            self.load_checkpoint(config.pop(LOAD_FROM_CONFIG_KEY, (None, None)))

            super().__init__(observation_space, action_space, config)

        def compute_actions(self,
                            obs_batch,
                            state_batches=None,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):
            actions, state_out, extra_fetches = self.tune_trainer.compute_actions(self.policy_id, obs_batch)
            return actions, state_out, extra_fetches

        def learn_on_batch(self, samples):
            raise NotImplementedError("FreezedPolicyFromTuneTrainer policy can't be trained")

        def get_weights(self):
            return {"checkpoint_path": self.checkpoint_path,
                    "policy_id": self.policy_id}
            # raise NotImplementedError("Not supported. Using tune_trainer.load_checkpoint in __init__ to load weights")

        def set_weights(self, weights):
            checkpoint_path = weights["checkpoint_path"]
            policy_id = weights["policy_id"]
            self.load_checkpoint((checkpoint_path, policy_id))
            # raise NotImplementedError("Not supported. Using tune_trainer.load_checkpoint in __init__ to load weights")

        def load_checkpoint(self, checkpoint_tuple):
            self.checkpoint_path, self.policy_id = checkpoint_tuple
            if self.checkpoint_path is not None:
                self.tune_trainer.load_checkpoint(self.checkpoint_path)

    return FreezedPolicyFromTuneTrainer