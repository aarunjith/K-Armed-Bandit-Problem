import numpy as np
import random
import math

class Agent:
    """
    Agent to play the bandits
    args :
        eps = Probability of exploration
        alpha = Learning rate of the agent
        verbose = verbosity of the agent
        character = Value update method of the agent, in 'eps_greedy', 'ucb', 'optimistic'
    """
    def __init__(self, eps=1, alpha=1, verbose=False, character='eps_greedy', c=2):
        self.eps = eps
        self.alpha = alpha
        self.rewards = list()
        self.verbosity = verbose
        self.character = character
        self.c = c
        assert (self.character in ['eps_greedy', 'ucb', 'optimistic'])

    def take_action(self, bandits, values, timestep):

        if self.character == 'eps_greedy':
            if np.random.rand() < self.eps:
                if self.verbosity:
                    print("Exploring the environment")
                i, bandit = random.choice(list(enumerate(bandits)))
                r = bandit.draw()
            else:
                if self.verbosity:
                    print("Greedy approach")
                i = np.argmax(values)
                bandit = bandits[i]
                r = bandit.draw()
            self.rewards.append(r)
            return i, r

        elif self.character == 'ucb':
            cbs = [self.c*b.get_conf_bound(timestep) + value for b, value in zip(bandits, values)]
            i = np.argmax(cbs)
            bandit = bandits[i]
            r = bandit.draw()
            self.rewards.append(r)
            return i, r

        else:
            if self.verbosity:
                print("Optimistic greedy approach")
            i = np.argmax(values)
            bandit = bandits[i]
            r = bandit.draw()
            self.rewards.append(r)
            return i, r






