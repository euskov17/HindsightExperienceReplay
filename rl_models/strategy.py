import random 
from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def sample(self, num_episode, achived_goals):
        pass

class FutureStrategy(Strategy):
    def __init__(self, n_samples=4):
        self.n_samples = n_samples
        
    def sample(self, num_episode, achived_goals):
        return random.choices(achived_goals[num_episode:], k=self.n_samples)

class FinalStrategy(Strategy):
    def sample(self, num_episode, achived_goals):
        return [achived_goals[-1]]

class RandomStrategy(Strategy):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self, num_episode, achived_goals):
        return random.choices(achived_goals, k=self.n_samples)
