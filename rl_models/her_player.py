import numpy as np
import torch
import torch.nn as nn
import random

from remember import HER, SimpleRecord

class Player:
    def __init__(self, env, agent, buffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def play_episode(self, epsilon=.1, max_steps=50):
        episode = []
        achived_goals = []
        state = self.env.reset()[0]
        desired_goal = state['desired_goal']
        
        goal = torch.tensor(desired_goal, dtype=torch.float)
            
        for step in range(max_steps):
            obs = torch.tensor(state['observation'], dtype=torch.float)
            stategoal = torch.cat([obs, goal], -1)
            
            if np.random.rand() < epsilon:
                action = torch.tensor(self.env.action_space.sample(), dtype=torch.float)
            else:
                action = self.agent.choose_action(stategoal, train=True)

            next_state, reward, done, _, info = self.env.step(action.numpy())
            score += reward
                
            next_obs = torch.tensor(next_state['observation'], dtype=torch.float)
            achived_goal = torch.tensor(next_state['achieved_goal'], dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)

            episode.append((obs, action, reward, next_obs, done, info))
            achived_goals.append(achived_goal)

            state = next_state
            if done:
                break

        success = info['is_success']
        return trajectory, goal, achived_goals, score, success

    def play_and_record(self, using_her=True, epsilon=.1, 
                        num_episodes=16, max_steps=50):
        success_rate = 0.0
        score_stat = 0.0
        record = HER() if using_her else SimpleRecord()
        for _ in range(num_episodes):
            trajectory, desired_goal, achived_goals, score, success = \
                self.play_episode(epsilon=epsilon, max_steps=max_steps)
        
            record.record(trajectory, desired_goal, achived_goals, self.buffer,
                        self.env.compute_reward)

            success_rate += success
            score_stat += score

        return score_stat / num_episodes, success_rate / num_episodes

    def update_agent(self, learning_freq=40, batch_size=128):
        for _ in range(learning_freq):
            batch = self.buffer.sample(batch_size)
            self.agent.learning_step(batch)

        self.agent.update_network_parameters()
