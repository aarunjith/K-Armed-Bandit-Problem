import numpy as np
import random


class Agent:
    """
    Agent to play the bandits
    args :
        eps = Probability of exploration
    """
    def __init__(self, eps=1, alpha=1, verbose=False):
        self.eps = eps
        self.alpha = alpha
        self.rewards = list()
        self.verbosity = verbose

    def take_action(self, bandits, values):
        if np.random.rand() < self.eps:
            if self.verbosity:
                print("Exploring the environment")
            i, bandit = random.choice(list(enumerate(bandits)))
            r = bandit.draw()
        else:
            if self.verbosity:
                print("Greedy Approach")
            i = np.argmax(values)
            bandit = bandits[i]
            r = bandit.draw()
        self.rewards.append(r)
        return i, r
