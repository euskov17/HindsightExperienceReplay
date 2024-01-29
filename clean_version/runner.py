import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from models.soft_actor_critic import SoftActorCritic
from models.ddpg import DDPG
# from rl_models.her import HindsightExperienceReplay
from her import HindsightExperienceReplay
from rl_models.replay_buffer import ReplayBuffer
from rl_models.utils import plot_learning_curve


class Runner:
    def __init__(self, *, env_name, max_episode_steps=50,
                 batch_size=128, using_her=True, gamma=.98,
                 one_goal=False,  buffer_size=int(1e6), 
                 tau=.05, agent_name='DDPG', hidden_size=64,
                 lr=1e-3, num_episodes=16, n_cycles=50,
                 name="DDPG"):
        self.env = gym.make(env_name, max_episode_steps=max_episode_steps)
        state_dim = self.env.observation_space['observation'].shape[0]        
        goal_dim = self.env.observation_space['desired_goal'].shape[0]
        actions_dim = self.env.action_space.shape[0]
        model = DDPG if agent_name == "DDPG" else SoftActorCritic
        self.agent = model(state_dim + goal_dim, actions_dim, hidden_size=64, lr=1e-3,
              tau=0.05, gamma=gamma) 
        self.buffer = ReplayBuffer(buffer_size)
        desired_goal = self.env.reset()[0]['desired_goal'] if one_goal else None
        self.her = HindsightExperienceReplay(self.env, self.agent, self.buffer,
                                             batch_size=batch_size, max_steps=max_episode_steps,
                                             one_goal_task=desired_goal)
        self.n_cycles = n_cycles
        self.num_episodes = num_episodes
        self.name = name

        self.success_history = []

    def run_epoch(self):
        success_history = []
        for _ in tqdm(range(self.n_cycles)):
            score, success = self.her.play_and_learn(num_episodes=self.num_episodes)
            success_history.append(success)
        rate = np.mean(success_history)
        self.success_history.append(rate)
        return rate
    
    def run_tests(self, n_tests):
        rate = self.her.test(n_tests)
        self.success_history.append(rate)
        return rate

    def plot(self):
        plot_learning_curve(len(self.success_history), self.success_history,
                            name=self.name)
        

class MultipleRunner:
    def __init__(self, configs):
        self.n_runners = len(configs)
        self.runners = [Runner(**config) for config in configs]
        self.rates = []

    def run_epoch(self):
        self.rates = {runner.name : runner.run_epoch() for runner in self.runners}

        # for runner in self.runners:
        #     runner.run_epoch()
        
    def run_tests(self, n_tests):
        self.rates = {runner.name : runner.run_tests(n_tests) for runner in self.runners}


    def plot(self):
        for runner in self.runners:
            runner.plot()

        for name, rate in self.rates.items():
            print(f"{name}  success rate {rate}")