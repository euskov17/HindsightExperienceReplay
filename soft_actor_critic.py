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

        sigma = torch.clamp(sigma, min=self.min_value, max=self.max_value)
        sigma = torch.nn.functional.softplus(sigma)
        return mu, sigma


    def apply(self, states, reparametrize=True):
        mu, sigma = self.get_mu_sigma(states)
        distribution = torch.distributions.Normal(mu, sigma)
        
        if reparametrize:
            u = distribution.rsample()
        else:
            u = distribution.sample()

        actions = torch.tanh(u) * self.scale_action

        log_prob = distribution.log_prob(actions)

        log_prob -= torch.log(1 + self.epsilon - actions.pow(2))
        log_prob = log_prob.sum(-1, keepdim=True)
        return actions, log_prob  

    def get_action(self, states):
        with torch.no_grad():
            states_torch = torch.Tensor(states)
            actions = self.apply(states_torch)[0].numpy()
        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoints_file))

class SoftActorCritic:
    def __init__(self, state_dim, action_dim, hidden_size=128, scale_action=1,
                 *, device=torch.device('cpu'), alpha=3e-4, lr=3e-4, tau=5e-4, gamma=0.99,
                 max_grad_norm=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.scale_action = scale_action
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = Actor(state_dim, action_dim, hidden_size, scale_action, name='actor').to(device)
        self.twin_critic = TwinCritic(state_dim, action_dim, hidden_size, lr=lr).to(device)
        self.value = Value(state_dim, hidden_size, name='value').to(device)
        self.target_value = Value(state_dim, hidden_size, name='target_value').to(device)
        self.target_value.eval()

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=lr)

    def choose_action(self, observation):
        if len(observation.shape) == 1:
            observation = [observation]
        state = torch.tensor(observation, device=self.device, dtype=torch.float)
        return self.actor.get_action(state).squeeze(-1)
    
    def __update_network_parameters(self):
        for param, target_param in zip(self.value.parameters(), self.target_value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self):
        for model in [self.actor, self.value, self.target_value, 
                        self.twin_critic #self.critic1, self.critic2
                        ]:
            model.save_checkpoint()

    def load_checkpoint(self):
        for model in [self.actor, self.value, self.target_value, 
                      self.twin_critic #self.critic1, self.critic2
                      ]:
            model.load_checkpoint()

    def __compute_target_value(self, states, reparametrize=True):
        actions, log_probs = self.actor.apply(states, reparametrize=reparametrize)
        log_probs = log_probs.view(-1)

        # q1_values = self.critic1(states, actions)
        # q2_values = self.critic2(states, actions)
        # critic_value = torch.min(q1_values, q2_values)

        critic_value = self.twin_critic.get_critics_min(states, actions)
        critic_value = critic_value.view(-1)
        return critic_value - self.alpha * log_probs


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
        dones = torch.tensor(dones, device=self.device, dtype=torch.int)

        values = self.value(states).view(-1)
        target_values = self.target_value(next_states).view(-1)
        target_values[dones] = 0.0

        target_value = self.__compute_target_value(states, reparametrize=False)

        value_loss = F.mse_loss(values, target_value)
        self.__optimize(self.value, self.optimizer_value, value_loss)

        actor_loss = -self.__compute_target_value(states, reparametrize=True).mean()
        self.__optimize(self.actor, self.optimizer_actor, actor_loss)

        q_target = rewards + self.gamma * target_values
        self.twin_critic.optimize_critics(states, actions, q_target)
        # # print(f"states_shape = {states.shape}, actions_shape = {actions.shape}")
        # q1_values = self.critic1(states, actions).view(-1)
        # q2_values = self.critic2(states, actions).view(-1)
        # critic1_loss = F.mse_loss(q1_values, q_target)
        # critic2_loss = F.mse_loss(q2_values, q_target)
        # self.__optimize(self.critic1, self.optimizer_critic1, critic1_loss)
        # self.__optimize(self.critic2, self.optimizer_critic2, critic2_loss)

        self.__update_network_parameters()