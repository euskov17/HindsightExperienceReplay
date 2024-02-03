import numpy as np
import torch
import torch.nn as nn
import random

from .remember import HER, SimpleRecorder

class Player:
    def __init__(self, env, agent, buffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def play_test(self, n_tests, max_steps=50):
        results = np.array([self.play_episode(train=False, max_steps=max_steps)
                            for _ in range(n_tests)])
        return tuple(np.mean(results, 0))

    def play_episode(self, epsilon=.3, max_steps=50, train=True):
        trajectory = []
        achived_goals = []
        state = self.env.reset()[0]
        desired_goal = state['desired_goal']
        score = 0.0
        goal = torch.tensor(desired_goal, dtype=torch.float)
            
        for step in range(max_steps):
            obs = torch.tensor(state['observation'], dtype=torch.float)
            stategoal = torch.cat([obs, goal], -1)
            
            if train and np.random.rand() < epsilon:
                action = torch.tensor(self.env.action_space.sample(), dtype=torch.float)
            else:
                action = self.agent.choose_action(stategoal, train=True)

            next_state, reward, done, _, info = self.env.step(action.numpy())
            score += reward
            state = next_state
            
            if train:
                next_obs = torch.tensor(next_state['observation'], dtype=torch.float)
                achived_goal = torch.tensor(next_state['achieved_goal'], dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)

                trajectory.append((obs, action, reward, next_obs, done, info))
                achived_goals.append(achived_goal)

            if done:
                break

        success = info['is_success']
        if not train:
            return score, success
        return trajectory, goal, achived_goals, score, success
        

    def play_and_record(self, using_her=True, epsilon=.1, 
                        num_episodes=16, max_steps=50,
                        recorder=None):
        success_rate = 0.0
        score_stat = 0.0
        
        if recorder is None:
            recorder = HER() if using_her else SimpleRecorder()

        for _ in range(num_episodes):
            trajectory, desired_goal, achived_goals, score, success = \
                self.play_episode(epsilon=epsilon, max_steps=max_steps)
        
            recorder.record(trajectory, desired_goal, achived_goals, self.buffer,
                        self.env.compute_reward)

            success_rate += success
            score_stat += score

        return score_stat / num_episodes, success_rate / num_episodes

    def update_agent(self, learning_freq=40, batch_size=128):
        for _ in range(learning_freq):
            batch = self.buffer.sample(batch_size)
            self.agent.learning_step(batch)

        self.agent.update_network_parameters()
