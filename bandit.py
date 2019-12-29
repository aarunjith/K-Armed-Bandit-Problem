import numpy as np


class Bandit:
    """
    Define a one armed bandit with a specified mean and standard deviation (Normal Distribution)
        Args :
            mean  = mean of the reward distribution
            sd = standard deviation of the reward distribution
    """
    def __init__(self, mean=0, sd=1, stationary=False):
        self.__mean = mean
        self.__sd = sd
        self.stationary = stationary

    def draw(self):
        self.check_props()
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

