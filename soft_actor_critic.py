import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Value, Actor, TwinCritic

class SoftActorCritic:
    def __init__(self, state_dim, action_dim, hidden_size=128, scale_action=1,
                 *, device=torch.device('cpu'), alpha=0.2, lr=3e-4, tau=5e-4, gamma=0.99,
                 max_grad_norm=10):
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
        self.optimizer_critic = torch.optim.Adam(self.twin_critic.parameters(), lr=lr)

    def choose_action(self, observation):
        if len(observation.shape) == 1:
            observation = [observation]
        state = torch.tensor(observation, device=self.device, dtype=torch.float)
        return self.actor.get_action(state).squeeze(-1)
    
    def __update_network_parameters(self):
        for param, target_param in zip(self.value.parameters(), self.target_value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self):
        for model in [self.actor, self.value, self.target_value, self.twin_critic ]:
            model.save_checkpoint()

    def load_checkpoint(self):
        for model in [self.actor, self.value, self.target_value, 
                      self.twin_critic]:
            model.load_checkpoint()

    def __compute_target_value(self, states, reparametrize=True):
        actions, log_probs = self.actor.apply(states, reparametrize=reparametrize)
        log_probs = log_probs.view(-1)

        critic_value = self.twin_critic.get_critics_min(states, actions)
        critic_value = critic_value.view(-1)
        return critic_value - log_probs

    def register_hook(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.register_hook(hook_fn)

    def __optimize(self, model, optimizer, loss):
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()
    

    def learning_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
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


        # values = self.value(states).view(-1)
        # target_values = self.target_value(next_states).view(-1)
        # target_values[dones] = 0.0
        # choosed_actions, log_probs = self.actor.apply(states)
        # critic_values = self.twin_critic.get_critics_min(states, actions).view(-1)
        # choosed_actions = choosed_actions.view(-1)
        # log_probs = log_probs.view(-1)


        # q_target = rewards + self.gamma * target_values
        # self.twin_critic.optimize_critics(states, actions, q_target)

        # self.optimizer_value.zero_grad()
        # target_value = (critic_values - log_probs).detach()
        # value_loss = F.mse_loss(values, target_value)
        # value_loss.backward()
        # self.optimizer_value.step()

        # self.optimizer_actor.zero_grad()
        # actor_loss = log_probs - critic_values.detach()
        # actor_loss = actor_loss.mean()
        # actor_loss.backward()
        # self.optimizer_actor.step()

        self.__update_network_parameters()