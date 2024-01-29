import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()        

        self.model = nn.Sequential(
            # nn.LayerNorm(state_dim + action_dim),
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, states, actions):
        sa = torch.cat([states, actions], dim=-1)
        qvalues = self.model(sa).squeeze()    
        return qvalues  
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64,
                 sigma=.2, max_action=1.0):
        super().__init__()        

        self.model = nn.Sequential(
            # nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.sigma = sigma
        
        self.low = -max_action
        self.high = max_action
        self.max_action = max_action
        self.noise = torch.distributions.Normal(torch.zeros(action_dim), sigma)
    
    def forward(self, states):
        return self.model(states)
    
    def get_best_actions(self, states, noise=False):
        actions = self.model(states) * self.max_action
        if noise:
            actions += self.noise.sample().squeeze(0)
        return actions.clamp(min=self.low, max=self.high) 
    

class DDPG:
    def __init__(self, state_dim, action_dim, hidden_size=64, 
                 *, device=torch.device('cpu'), alpha=0.2, lr=1e-3, tau=.5, gamma=0.99,
                 max_grad_norm=10, sigma=0.2,
                 max_action=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.action_low = -max_action
        self.action_high = max_action
        # self.noise = torch.distributions.Normal(torch.zeros(action_dim), torch.eye(action_dim) * sigma)

        self.actor = Actor(state_dim, action_dim, hidden_size, 
                            max_action=max_action, sigma=sigma)
        self.target_actor = deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic = deepcopy(self.critic)
       
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def choose_action(self, observation, train=False):
        with torch.no_grad():
            return self.actor.get_best_actions(observation, noise=train)

    def __update_network_parameters(self, model, target_model):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_network_parameters(self):
        self.__update_network_parameters(self.actor, self.target_actor)
        self.__update_network_parameters(self.critic, self.target_critic)
        

    def save_checkpoint(self):
        for model in [self.actor, self.critic1, self.critic2, self.target_critic1,
                      self.target_critic2]:
            model.save_checkpoint()

    def load_checkpoint(self):
        for model in [self.actor, self.critic1, self.critic2, self.target_critic1,
                      self.target_critic2]:
            model.load_checkpoint()

    def __optimize(self, model, optimizer, loss):
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()
    

    def learning_step(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        with torch.no_grad():
            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            next_states = torch.stack(next_states)
            not_dones = ~torch.tensor(dones, dtype=torch.bool)

        q_values = self.critic(states, actions)

        with torch.no_grad():
            target_actions = self.target_actor.get_best_actions(next_states)
            next_state_values = self.target_critic(next_states, target_actions)
            
            q_target = rewards + not_dones * self.gamma * next_state_values
            q_target = torch.clamp(q_target, -1.0 / (1.0 - self.gamma), 0)

        qloss = F.mse_loss(q_values, q_target)
        self.__optimize(self.critic, self.critic_optimizer, qloss)

        best_actions = self.actor.get_best_actions(states)
        policy_loss = -self.critic(states, best_actions) + (best_actions ** 2).mean()
        self.__optimize(self.actor, self.actor_optimizer, policy_loss)
