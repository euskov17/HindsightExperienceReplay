import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_networks import Critic, MLPLayer

class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, 
                 min_value=-2, max_value=20, epsilon=1e-5):
        super().__init__()        
        
        self.model = MLPLayer(state_dim, hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.sigma = nn.Linear(hidden_size, action_dim)

        self.min_value = min_value
        self.max_value = max_value 
        self.epsilon = epsilon
    
    def apply(self, states): 
        out = self.model(states)
        mu = self.mu(out)
        sigma = self.sigma(out)

        sigma = torch.clamp(sigma, min=self.min_value, max=self.max_value)
        sigma = F.softplus(sigma)
        
        distribution = torch.distributions.Normal(mu, sigma)
        
        u = distribution.rsample()
        actions = torch.tanh(u)

        log_prob = distribution.log_prob(u)

        log_prob -= torch.log(1 + self.epsilon - actions.pow(2))
        log_prob = log_prob.sum(-1, keepdim=True)
        return actions, log_prob.view(-1)

    def get_action(self, states):
        with torch.no_grad():
            states_torch = torch.Tensor(states)
            actions = self.apply(states_torch)[0].clamp(-1, 1)
            return actions    
        

class SoftActorCritic:
    def __init__(self, state_dim, action_dim, hidden_size=128, scale_action=1,
                 *, device=torch.device('cpu'), alpha=0.2, lr=5e-4, tau=.05, gamma=0.99,
                 max_grad_norm=10):
        self.scale_action = scale_action
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = SAC_Actor(state_dim, action_dim, hidden_size).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_size).to(device)

        self.target_critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_size).to(device) 
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=lr)

    def choose_action(self, observation, train=True):
        return self.actor.get_action(observation)
    
    def __update_network_parameters(self, model, target_model):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_network_parameters(self):
        self.__update_network_parameters(self.critic1, self.target_critic1)
        self.__update_network_parameters(self.critic2, self.target_critic2)

    def save_checkpoint(self):
        for model in [self.actor, self.critic1, self.critic2, self.target_critic1,
                      self.target_critic2]:
            model.save_checkpoint()

    def load_checkpoint(self):
        for model in [self.actor, self.critic1, self.critic2, self.target_critic1,
                      self.target_critic2]:
            model.load_checkpoint()

    def __compute_critic_value(self, rewards, next_states, not_done, reparametrize=True):
        with torch.no_grad():
            actions, log_probs = self.actor.apply(next_states)
            q1 = self.critic1(next_states, actions)
            q2 = self.critic2(next_states, actions)
            critic_target = rewards + self.gamma * not_done * torch.min(q1, q2)
            critic_target -= self.alpha * log_probs        
        return critic_target 

    def __compute_actor_loss(self, states):
        actions, log_probs = self.actor.apply(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        actor_loss = -torch.min(q1, q2) + self.alpha * log_probs
        return actor_loss.mean()

    def __optimize(self, model, optimizer, loss):
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
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
  

        target_critic = self.__compute_critic_value(rewards, next_states, not_dones)
        critic1_loss = F.mse_loss(self.critic1(states, actions), target_critic)
        self.__optimize(self.critic1, self.optimizer_critic1, critic1_loss)

        critic2_loss = F.mse_loss(self.critic2(states, actions), target_critic)
        self.__optimize(self.critic2, self.optimizer_critic2, critic2_loss)

        actor_loss = self.__compute_actor_loss(states)
        self.__optimize(self.actor, self.optimizer_actor, actor_loss)
