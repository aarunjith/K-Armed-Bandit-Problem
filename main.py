from agent import Agent
from bandit import Bandit
import numpy as np

class Env:
    """
    Environment with k-bandits where the agent plays
    args:
        k = Number of bandits
        stationary = Property of the bandit
            True : Change the reward distribution randomly each time
            False : Reward distribution is static
    """
    def __init__(self, k=10, stationary=True):
        print(f'Initialising {k} Bandits')
        self.k = k
        self.timestep = 0
        self.stationary = stationary
        self.values = np.zeros(self.k)
        self.updates = np.zeros(self.k)
        mu = [np.random.randint(10) for _ in range(self.k)]
        sig = [np.random.rand() for _ in range(self.k)]
        print(f"Initial actual average rewards are {mu}")
        self.bandits = [Bandit(mean, sd, stationary) for (mean, sd) in zip(mu, sig)]

    def update_values(self, agent):
        index, reward = agent.take_action(self.bandits, self.values)
        self.timestep += 1
        self.updates[index] += 1
        if self.stationary:
            self.values[index] += 1/self.timestep*(reward - self.values[index])
        else:
            self.values[index] += agent.alpha*(reward - self.values[index])


if __name__ == "__main__":
    import tqdm
    TIMESTEPS = 1000000
    env = Env(10, True)
    agent = Agent(eps=0.01, alpha=0.1)
    for t in tqdm.tqdm(range(TIMESTEPS)):
        env.update_values(agent)
    print(f"Final actual average rewards are {[b.get_average_reward() for b in env.bandits]} ")
    print(f"Final bandit values : {env.values}")
    print(f"Number of Updates : {env.updates}")
    print(f"Average reward obtained : {sum(agent.rewards)/TIMESTEPS}")


