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

class HER:
    def __init__(self, strategy=FutureStrategy()):
        self.strategy = strategy

    def record(self, trajectory, achived_goals, buffer, compute_reward):
        steps = len(trajectory)
        n = np.random.randint(0, steps)
        idxes = np.random.randint(steps, size=n)
        # idxes = range(steps)
        for idx in idxes:
            additional_goals = self.strategy.sample(idx, achived_goals)
            obs, action, reward, next_obs, done, info = trajectory[idx]
            achived_goal = achived_goals[idx]

            for current_goal in additional_goals:
                reward = torch.tensor(compute_reward(
                                      achived_goal, 
                                      current_goal, info))
                done = reward != -1.0
                # print(f"Reward {reward}, done {done}")
                stategoal = torch.cat([obs, current_goal], -1)
                nextgoal = torch.cat([next_obs, current_goal], -1)
                buffer.add(stategoal, action, reward, nextgoal, done)


class Player:
    def __init__(self, env, agent, buffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def play_and_record(self, using_her=True, epsilon=.1, 
                        num_episodes=16, max_steps=50):
        success = 0.0
        score = 0.0
        her = HER() if using_her else None
        for _ in range(num_episodes):
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
                    # print(action)
                else:
                    action = self.agent.choose_action(stategoal, train=True)
                    # print(action)
                
                next_state, reward, done, _, info = self.env.step(action.numpy())
                score += reward
                
                next_obs = torch.tensor(next_state['observation'], dtype=torch.float)
                achived_goal = torch.tensor(next_state['achieved_goal'], dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)
                # done = bool(info['is_success'])

                episode.append((obs, action, reward, next_obs, done, info))
                achived_goals.append(achived_goal)

                nextgoal = torch.cat([next_obs, goal], -1)
                self.buffer.add(stategoal, action, reward, nextgoal, done)

                state = next_state
                # if info['is_success']:
                #     print(reward, done, self.env.compute_reward(achived_goal, goal, info))
                    # print("it happens")
                # if done != info['is_success']:
                #     print(f"Some Error {done}  {tr} {info['is_success']} {torch.norm(goal - achived_goal, 2)}")
                if done:
                    # success += 1
                    break

            success += info['is_success']
            if not using_her:
                continue
        
            her.record(episode, achived_goals, self.buffer,
                        self.env.compute_reward)
        
        return score / num_episodes, success / num_episodes

    def update_agent(self, learning_freq=40, batch_size=128):
        for _ in range(learning_freq):
            batch = self.buffer.sample(batch_size)
            self.agent.learning_step(batch)

        self.agent.update_network_parameters()