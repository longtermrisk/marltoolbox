import numpy as np
from gym.core import Wrapper

class RewardUncertaintyWrapper(Wrapper):

    def __init__(self, env, reward_uncertainty_std, reward_uncertainty_mean=0.0):
        assert reward_uncertainty_std >= 0.0
        self.reward_uncertainty_std = reward_uncertainty_std
        self.reward_uncertainty_mean = reward_uncertainty_mean
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward_dict):
        for k in reward_dict.keys():
            reward_dict[k] = reward_dict[k] + np.random.normal(loc=self.reward_uncertainty_mean,
                                                               scale=self.reward_uncertainty_std, size=1)
        return reward_dict

