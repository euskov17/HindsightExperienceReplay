import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FutureStrategy:
    def __init__(self, n_samples=4):
        self.n_samples = n_samples

    def sample(self, num_episode, achived_goals):
        # sz = min(len(achived_goals) - num_episode, self.n_samples)
        return random.choices(achived_goals[num_episode:], k=self.n_samples) 
    
class FinalStrategy:
    def sample(self, num_episode, achived_goals):
        return achived_goals[-1]
    
class RandomStrategy:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, num_episode, achived_goals):
        return random.sample(achived_goals, k=self.num_samples)

class HindsightExperienceReplay:
    def __init__(self, env, agent, buffer, strategy=FutureStrategy(), 
                 max_steps=64, batch_size=64, learning_freq=40):
        self.env = env
        self.agent = agent
        self.strategy = strategy
        self.buffer = buffer
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_freq = learning_freq

    def play_and_learn(self):
        state = self.env.reset()[0]
        goal = state['desired_goal']
        episode = []
        achived_goals = []
        score = 0.0
        success = False

        for _ in range(self.max_steps):
            stategoal = np.concatenate([state['observation'], goal], -1)
            action = self.agent.choose_action(stategoal)
            next_state, reward, done, _, _ = self.env.step(action)
            score += reward
            
            episode.append((state['observation'], action, next_state['observation'], done))
            achived_goals.append(next_state['achieved_goal'])

            nextgoal = np.concatenate([next_state['observation'], goal], -1)
            self.buffer.add(stategoal, action, reward, nextgoal, done)
            if done:
                success = True
                break

            state = next_state

        for i in range(len(episode)):
            additional_goals = self.strategy.sample(i, achived_goals)
            obs, action, next_obs, done = episode[i]
            achived_goal = achived_goals[i]

            #add achived goal
            # sa = np.concatenate([obs, achived_goal], -1)
            # na = np.concatenate([next_obs, achived_goal], -1)
            # self.buffer.add(sa, action, 0.0, na, True)
            #add another goals

            for current_goal in additional_goals:
                done = np.allclose(achived_goal, current_goal)
                reward = 0 if done else -1
                stategoal = np.concatenate([obs, current_goal], -1)
                nextgoal = np.concatenate([next_obs, current_goal], -1)
                self.buffer.add(stategoal, action, reward, nextgoal, done)

        if len(self.buffer) < self.batch_size:
            return score, success

        for i in range(self.learning_freq):
            batch = self.buffer.sample(self.batch_size)
            self.agent.learning_step(batch)

        return score, success