import torch
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from abc import ABC

from ..rl_models.her_player import Player
from ..models.ddpg import DDPG
from ..models.soft_actor_critic import SoftActorCritic
from ..rl_models.utils import plot_learning_curve
from ..rl_models.utils import ReplayBuffer


class TestRunner(ABC):
    @abstract_method
    def RunEpoch(self, n_cycles, n_episodes, batch_size, learning_freq, epsilon):
        pass

    @abstract_method
    def RunTest(self, n_epochs, n_cycles, n_episodes, batch_size, learning_freq, epsilon,
            running):
        pass


class OneTestRunner(TestRunner):
    def __init__(self, env, name="DDPG with HER", agent, using_her=True, *, 
            max_episode_step=50, buffer_size=int(1e6)):
        self.env = env
        self.agent = agent 
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.player = Player(self.env, self.agent, self.replay_buffer)

        self.name = name
        self.using_her = using_her

    def RunEpoch(self, n_cycles=50, n_episodes=16, 
            batch_size=128, learning_freq=40, epsilon=.3):
        
        score_history = []
        success_history = []

        for _ in tqdm(n_cycles):
            score, success = self.player.play_and_record(n_episodes=n_episodes,
                    epsilon=epsilon, using_her=self.using_her)

            player.update_agent()
            score_history.append(score)
            success_history.append(success)

        success_rate = np.mean(success_history)
        score_stat = np.mean(score_history)
        return score_stat, success_rate

    def RunTest(self, n_epochs=100, n_cycles=50, n_episodes=16, 
            batch_size=128, learning_freq=40, epsilon=.3, running=False):
        score_history = []
        success_history = []
        
        for epoch in range(n_epochs):
            score_stat, success_rate = self.RunEpoch(n_cycles, n_episodes, batch_size,
                    learning_freq, epsilon)
            score_history.append(score_stat)
            success_history.append(success_rate)

            plot_learning_curve(success_history, name=self.name, running=running)
            print(f"Epoch {epoch + 1} success_rate {success_rate}")


class MultipleTestRunner(TestRunner):
    
