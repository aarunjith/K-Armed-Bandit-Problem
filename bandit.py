import numpy as np
import math


class Bandit:
    """
    Define a one armed bandit with a specified mean and standard deviation (Normal Distribution)
        Args :
            mean  = mean of the reward distribution
            sd = standard deviation of the reward distribution
            stationary = Boolean
                True : Reward distributions change over timesteps
                False : Otherwise
    """
    def __init__(self, mean=0, sd=1, stationary=False):
        self.__mean = mean
        self.__sd = sd
        self.stationary = stationary
        self.updates = 0

    def draw(self):
        self.check_props()
        self.updates += 1
        return np.random.normal(self.__mean, self.__sd)

    def set_properties(self, mean, sd):
        self.__mean = mean
        self.__sd = sd

    def check_props(self):
        if not self.stationary:
            n_mean = np.random.randint(0, 5)
            n_sd = np.random.rand()
            self.set_properties(n_mean, n_sd)

    def get_average_reward(self):
        return self.__mean

    def get_conf_bound(self, timestep):
        if self.updates == 0:
            return float('inf')
        else:
            return math.sqrt(math.log(timestep, math.exp(1))/self.updates)
