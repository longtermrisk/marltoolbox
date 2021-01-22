import gym
import numpy as np

# from gym.spaces import prng
from gym.utils import seeding


class OneHot(gym.Space):
    """
    One-hot space. Used as the observation space.
    """
    def __init__(self, n):
        self.n = n
        # self.np_random = None

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample(self):
        return self.np_random.multinomial(1, [1. / self.n] * self.n)

    def contains(self, x):
        return isinstance(x, np.ndarray) and \
               x.shape == (self.n, ) and \
               np.all(np.logical_or(x == 0, x == 1)) and \
               np.sum(x) == 1

    @property
    def shape(self):
        return (self.n, )

    def __repr__(self):
        return "OneHot(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
