import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FutureStrategy:
    def __init__(self, n_samples=4):
        self.n_samples = n_samples

    def sample(self, num_episode, achived_goals):
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
                 max_steps=64, batch_size=64, learning_freq=40, train_start=1000):
        self.env = env
        self.agent = agent
        self.strategy = strategy
        self.buffer = buffer
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.train_start = train_start

    def play_and_learn(self, num_episodes=16, her=True):
        score = 0.0
        success = 0
        for _ in range(num_episodes):
            episode = []
            achived_goals = []
            state = self.env.reset()[0]
            goal = state['desired_goal']
            goal = torch.tensor(goal, dtype=torch.float)
            
            for step in range(self.max_steps):
                obs = torch.tensor(state['observation'], dtype=torch.float)
                stategoal = torch.cat([obs, goal], -1)
                action = self.agent.choose_action(stategoal)
                
                next_state, reward, done, _, _ = self.env.step(action.numpy())
                score += reward
                
                next_obs = torch.tensor(next_state['observation'], dtype=torch.float)
                achived_goal = torch.tensor(next_state['achieved_goal'], dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)

                episode.append((obs, action, reward, next_obs, done))
                achived_goals.append(achived_goal)
                state = next_state
                
                if done:
                    success += 1
                    break


            if not her:
                continue

            step_taken = step

            for i in range(step_taken):
                additional_goals = self.strategy.sample(i, achived_goals)
                obs, action, reward, next_obs, done = episode[i]
                achived_goal = achived_goals[i]

                #add achived goal
                sa = torch.cat([obs, goal], -1)
                na = torch.cat([next_obs, goal], -1)
                self.buffer.add(sa, action, reward, na, done)
                #add another goals

                for current_goal in additional_goals:
                    done = torch.equal(achived_goal, current_goal)
                    reward = torch.tensor(0.0 if done else -1.0, dtype=torch.float)
                    stategoal = torch.cat([obs, current_goal], -1)
                    nextgoal = torch.cat([next_obs, current_goal], -1)
                    self.buffer.add(stategoal, action, reward, nextgoal, done)


        if len(self.buffer) < self.train_start:
            return score, success / num_episodes

        for i in range(self.learning_freq):
            batch = self.buffer.sample(self.batch_size)
            self.agent.learning_step(batch)

        self.agent.update_network_parameters()

        return score / num_episodes, success / num_episodes