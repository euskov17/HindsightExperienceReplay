import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()        

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_qvalues(self, states, actions):
        sa = torch.cat([states, actions], dim=-1)
        qvalues = self.model(sa).squeeze()
        return qvalues
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, min_value=-2, max_value=20, epsilon=1e-5):
        super().__init__()        
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.model_mu = nn.Linear(hidden_dim, action_dim)

        self.model_sigma = nn.Linear(hidden_dim, action_dim)

        self.min_value = min_value
        self.max_value = max_value 
        self.epsilon = epsilon
    
    def apply(self, states):
        probs = self.mlp(states)
        mu = self.model_mu(probs)
        sigma = self.model_sigma(probs)

        sigma = torch.clamp(sigma, min=self.min_value, max=self.max_value)
        sigma = torch.nn.functional.softplus(sigma)
        
        distribution = torch.distributions.Normal(mu, sigma)
        
        u = distribution.rsample()
        actions = torch.tanh(u)

        log_prob = distribution.log_prob(actions)

        log_prob -= torch.log(1 + self.epsilon - actions.pow(2))
        log_prob = log_prob.sum(-1, keepdim=True)
        return actions, log_prob  

    def get_action(self, states):
        with torch.no_grad():
            states_torch = torch.Tensor(states)
            actions = self.apply(states_torch)[0].numpy()
        return actions
    

class SoftActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 *, device=torch.device('cpu'), alpha=3e-4, lr=3e-4, tau=5e-4, gamma=0.99,
                 max_grad_norm=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.alpha = alpha
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim) 
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)


        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())   
        
    def __update_target_network(self, model, target_model):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def __update_target_networks(self):
        self.__update_target_network(self.actor, self.target_actor)
        self.__update_target_network(self.critic1, self.target_critic1)
        self.__update_target_network(self.critic2, self.target_critic2)

    def choose_action(self, observation):
        state = torch.tensor([observation])
        return self.actor.get_action(state)
    
    def __compute_critic_target(self, rewards, next_states, dones):
        with torch.no_grad():
            actions, log_probs = self.actor.apply(next_states)
            q_values1 = self.target_critic1.get_qvalues(next_states, actions)
            q_values2 = self.target_critic2.get_qvalues(next_states, actions)
            critic_target = rewards + self.gamma * (1 - dones) * torch.min(q_values1, q_values2)

            critic_target -= self.alpha * log_probs
        return critic_target
    
    def __compute_actor_loss(self, states):
        actions, log_probs = self.actor.apply(states)
        q_values1 = self.critic1.get_qvalues(states, actions)
        q_values2 = self.critic2.get_qvalues(states, actions)
        loss = -torch.min(q_values1, q_values2) + self.alpha * log_probs
        return loss.mean()


    def __optimize(self, model, optimizer, loss):
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()

    def learning_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.tensor(states, device=self.device, dtype=torch.float) 
        actions = torch.tensor(actions, device=self.device, dtype=torch.float) 
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float) 
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float) 
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)

        critic1_q_values = self.critic1.get_qvalues(states, actions)
        critic2_q_values = self.critic2.get_qvalues(states, actions)
        critic_target = self.__compute_critic_target(rewards,  next_states, dones)
        
        critic1_loss = F.mse_loss(critic1_q_values, critic_target)
        critic2_loss = F.mse_loss(critic2_q_values, critic_target)

        self.__optimize(self.critic1, self.optimizer_critic1, critic1_loss)
        self.__optimize(self.critic2, self.optimizer_critic2, critic2_loss)

        actor_loss = self.__compute_actor_loss(states)
        self.__optimize(self.actor, self.optimizer_actor, actor_loss)
        
        self.__update_target_networks()