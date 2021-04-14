import numpy as np


def add_RewardUncertaintyEnvClassWrapper(
    env_class, reward_uncertainty_std, reward_uncertainty_mean=0.0
):
    class RewardUncertaintyEnvClassWrapper(env_class):
        def step(self, action):
            observations, rewards, done, info = super().step(action)
            return observations, self.reward_wrapper(rewards), done, info

        def reward_wrapper(self, reward_dict):
            for k in reward_dict.keys():
                reward_dict[k] += np.random.normal(
                    loc=reward_uncertainty_mean,
                    scale=reward_uncertainty_std,
                    size=(),
                )
            return reward_dict

    return RewardUncertaintyEnvClassWrapper
