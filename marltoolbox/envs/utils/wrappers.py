import numpy as np

def add_RewardUncertaintyEnvClassWrapper(EnvClass, reward_uncertainty_std, reward_uncertainty_mean=0.0):
    class RewardUncertaintyEnvClassWrapper(EnvClass):

        def step(self, action):
            print("in wrapper")
            observations, rewards, done, info = super().step(action)
            return observations, self.reward_wrapper(rewards), done, info

        def reward_wrapper(self, reward_dict):
            for k in reward_dict.keys():
                print("in reward_wrapper before", reward_dict[k])
                reward_dict[k] += np.random.normal(loc=reward_uncertainty_mean, scale=reward_uncertainty_std, size=())
                print("in reward_wrapper after", reward_dict[k])
            return reward_dict

    return RewardUncertaintyEnvClassWrapper

