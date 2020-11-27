
import gym

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import AgentID, ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict, Tuple, Union

from marltoolbox.utils.restore import  LOAD_FROM_CONFIG_KEY

class FreezedPolicyFromTuneTrainer(Policy):

    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, config: TrainerConfigDict):

        self.policy_id = config["policy_id"]
        # TuneTrainerClass = config.pop("TuneTrainerClass")
        TuneTrainerClass = config["TuneTrainerClass"]
        tune_trainer = TuneTrainerClass(config=config)
        # remove checkpoint from config to prevent after_init_load_checkpoint_from_config
        # to try to load a RLLib checkpoint
        # TODO make this cleaner, there should be no need to know about after_init_load_checkpoint_from_config,
        #  remove side effects
        checkpoint_path = config.pop(LOAD_FROM_CONFIG_KEY)
        tune_trainer.load_checkpoint(checkpoint_path)
        self.tune_trainer = tune_trainer

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
        raise NotImplementedError("Not supported. Using tune_trainer.load_checkpoint in __init__ to load weights")

    def set_weights(self, weights):
        raise NotImplementedError("Not supported. Using tune_trainer.load_checkpoint in __init__ to load weights")
