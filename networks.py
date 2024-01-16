import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, \
                name='critic', checkpoints_directory='chkpt_sac/'):
        super().__init__()        
        self.name = name
        self.checkpoints_directory = checkpoints_directory
        self.checkpoints_file = os.path.join(self.checkpoints_directory, name + '_sac')

        self.model = nn.Sequential(
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
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoints_file))

class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, lr=3e-4):
        super().__init__()
        self.critic1 = Critic(state_dim, action_dim, hidden_size, name='critic1')
        self.critic2 = Critic(state_dim, action_dim, hidden_size, name='critic2')
        self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=lr)

    def forward(self, states, actions):
        return self.critic1(states, actions), self.critic2(states, actions)

    def get_critics_min(self, states, actions):
        return torch.min(*self.forward(states, actions))

    def save_checkpoint(self):
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()

    def load_checkpoint(self):
        self.critic2.load_checkpoint()
        self.critic1.load_checkpoint()

    def optimize_critics(self, states, actions, target):
        self.optimizer_critic1.zero_grad()
        self.optimizer_critic2.zero_grad()
        
        q1_values = self.critic1(states, actions).view(-1)
        q2_values = self.critic2(states, actions).view(-1)

        critic1_loss = F.mse_loss(q1_values, target)
        critic2_loss = F.mse_loss(q2_values, target)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()

        self.optimizer_critic1.step()
        self.optimizer_critic2.step()


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=128, \
                 name='value', checkpoints_directory='chkpt_sac/'):
        super().__init__()        
        self.name = name
        self.checkpoints_directory = checkpoints_directory
        self.checkpoints_file = os.path.join(self.checkpoints_directory, name + '_sac')

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, states):
        return self.model(states)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoints_file))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, scale_action=1,
                 name='actor', checkpoints_directory='chkpt_sac/', 
                 *, min_value=-2, max_value=20, epsilon=1e-5):
        super().__init__()   
        self.scale_action = scale_action
        self.name = name
        self.checkpoints_directory = checkpoints_directory
        self.checkpoints_file = os.path.join(checkpoints_directory, name + '_sac')     
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.model_mu = nn.Linear(hidden_size, action_dim)

        self.model_sigma = nn.Linear(hidden_size, action_dim)

        self.min_value = min_value
        self.max_value = max_value 
        self.epsilon = epsilon
    
    def get_mu_sigma(self, states):
        mlp_out = self.mlp(states)
        mu = self.model_mu(mlp_out)
        sigma = self.model_sigma(mlp_out)

        # sigma = torch.clamp(sigma, min=self.min_value, max=self.max_value)
        # sigma = torch.clamp(sigma, min=self.epsilon, max=1)
        # sigma = F.softplus(sigma)
        # sigma = (self.min_value + 0.5 * (self.max_value - self.min_value) * (sigma + 1)).exp()
        sigma = torch.clamp(sigma, min = self.epsilon, max=1)
        return mu, sigma


    def apply(self, states, reparametrize=True):
        mu, sigma = self.get_mu_sigma(states)
        distribution = torch.distributions.Normal(mu, sigma)
        
        if reparametrize:
            u = distribution.rsample()
        else:
            u = distribution.sample()

        actions = torch.tanh(u) * self.scale_action

        log_prob = distribution.log_prob(u)

        log_prob -= torch.log(1 + self.epsilon - actions.pow(2))
        log_prob = log_prob.sum(-1, keepdim=True)
        return actions, log_prob  

    def get_action(self, states):
        states_torch = torch.Tensor(states)
        actions = self.apply(states_torch, False)[0].numpy()
        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoints_file))

