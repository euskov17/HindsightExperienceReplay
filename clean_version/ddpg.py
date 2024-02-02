import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import CriticNetwork, ActorNetwork

class DDPG:
    def __init__(self, state_dim, action_dim, hidden_size=128,
                 *, lr=1e-3, tau=.05, gamma=0.98,
                 max_grad_norm=10, sigma=0.2, max_action=1.0,
                 l2_reg=1.0, use_clip=False, clip_low=-1.0 / (1.0 - .98), clip_high=0.0):
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.action_low = -max_action
        self.action_high = max_action
        self.use_clip = use_clip
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.actions_dim = action_dim

        # self.noise = torch.distributions.Normal(torch.zeros(action_dim), torch.eye(action_dim) * sigma)
        self.sigma = sigma

        self.actor = ActorNetwork(state_dim, action_dim, hidden_size, max_action=max_action)
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_size, max_action=max_action)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(state_dim, action_dim, hidden_size)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_size)
        self.target_critic.load_state_dict(self.critic.state_dict())
       
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def choose_action(self, observation, train=False):
        with torch.no_grad():
            action = self.actor(observation)
            # print(f"Before {action}", end=' ')
            if train:
                action += torch.empty(self.actions_dim).normal_(mean=0, std=self.sigma) #self.noise.sample().squeeze(0)
            # print(f"After {action}")    
            return action.clamp(min=self.action_low, max=self.action_high)

    def __update_network_parameters(self, model, target_model):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_network_parameters(self):
        self.__update_network_parameters(self.actor, self.target_actor)
        self.__update_network_parameters(self.critic, self.target_critic)

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
            target_actions = self.target_actor(next_states)
            next_state_values = self.target_critic(next_states, target_actions)
            
        q_target = rewards + not_dones * self.gamma * next_state_values
        if self.use_clip:
            q_target = torch.clamp(q_target, self.clip_low, self.clip_high)

        qloss = F.mse_loss(q_values, q_target)
        self.__optimize(self.critic, self.critic_optimizer, qloss)

        best_actions = self.actor(states)
        policy_loss = -self.critic(states, best_actions) + self.l2_reg * (best_actions ** 2).mean()
        self.__optimize(self.actor, self.actor_optimizer, policy_loss)
