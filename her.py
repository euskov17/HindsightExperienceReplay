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
            
            for step in range(self.max_steps):
                stategoal = np.concatenate([state['observation'], goal], -1)
                action = self.agent.choose_action(stategoal)
                
                next_state, reward, done, _, _ = self.env.step(action)
                score += reward
                
                episode.append((state['observation'], action, reward, next_state['observation'], done))
                achived_goals.append(next_state['achieved_goal'])

                nextgoal = np.concatenate([next_state['observation'], goal], -1)
                self.buffer.add(stategoal, action, reward, nextgoal, done)

                state = next_state
                
                if done:
                    success += 1
                    break


            if not her:
                continue
            step_taken = step

            for i in range(step_taken):
                # additional_goals = self.strategy.sample(i, achived_goals)
                obs, action, reward, next_obs, done = episode[i]
                achived_goal = achived_goals[i]

                #add achived goal
                # sa = np.concatenate([obs, goal], -1)
                # na = np.concatenate([next_obs, goal], -1)
                # self.buffer.add(sa, action, reward, na, done)
                #add another goals

                for _ in range(4):
                    future = random.randint(i, step_taken)
                    new_goal = achived_goals[future]
                    done = np.allclose(achived_goal, new_goal)
                    reward = 0 if done else -1
                    stategoal = np.concatenate([obs, new_goal], -1)
                    nextgoal = np.concatenate([next_obs, new_goal], -1)
                    self.buffer.add(stategoal, action, reward, nextgoal, done)


                # for current_goal in additional_goals:
                #     done = np.allclose(achived_goal, current_goal)
                #     reward = 0 if done else -1
                #     stategoal = np.concatenate([obs, current_goal], -1)
                #     nextgoal = np.concatenate([next_obs, current_goal], -1)
                #     self.buffer.add(stategoal, action, reward, nextgoal, done)

        # for episode in range(num_episodes):
        #     # Run episode and cache trajectory
        #     episode_trajectory = []
        #     dct = self.env.reset()[0]
        #     goal = dct['desired_goal']
        #     state = dct['observation']
        #     for step in range(self.max_steps):

        #         state_ = np.concatenate((state, goal))
        #         action = self.agent.choose_action(state_)
        #         next_state, reward, done, _, _ = self.env.step(action)
        #         episode_trajectory.append((state, action, reward, next_state['observation'], done))
        #         state = next_state['observation']
        #         if done:
        #             successes += 1
        #             break

        #     # Fill up replay memory
        #     steps_taken = step
        #     for t in range(steps_taken):

        #         # Standard experience replay
        #         state, action, reward, next_state, done = episode_trajectory[t]
        #         state_, next_state_ = np.concatenate((state, goal)), np.concatenate((next_state, goal))
        #         self.buffer.add(state_, action, reward, next_state_, done)

        #         # Hindsight experience replay
        #         if not her:
        #             continue

        #         for _ in range(4):
        #             future = random.randint(t, steps_taken)  # index of future time step
        #             new_goal = episode_trajectory[future][3]  # take future next_state and set as goal
                    
        #             new_done = np.allclose(next_state, new_goal)
        #             new_reward = 0 if new_done else -1
        #             # new_reward, new_done = self.env.compute_reward(next_state, new_goal)
                    
        #             state_, next_state_ = np.concatenate((state, new_goal)), np.concatenate((next_state, new_goal))
        #             self.buffer.add(state_, action, new_reward, next_state_, new_done)



        if len(self.buffer) < self.train_start:
            return score, success / num_episodes

        for i in range(self.learning_freq):
            batch = self.buffer.sample(self.batch_size)
            self.agent.learning_step(batch)

        self.agent.update_network_parameters()

        return score / num_episodes, success / num_episodes