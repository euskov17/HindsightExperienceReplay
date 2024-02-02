import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from base_networks import MLPLayer

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.model = MLPLayer(input_dim, output_dim + 1, hidden_size)

    def forward(self, input):
        out = self.model(input)
        value = out[..., 0]
        advantage = out[..., 1:]
        advantage = advantage - advantage.mean(-1, keepdim=True)
        return value[..., None] + advantage

class DQN:
    def __init__(self, state_dim, n_actions, hidden_size=128, gamma=0.98,
                lr=1e-3):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma

        self.model = DuelingQNetwork(state_dim, n_actions, hidden_size)
        self.target_model = DuelingQNetwork(state_dim, n_actions, hidden_size)
        self.update_network_parameters()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon = 0.2

    def update_network_parameters(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, train=True):
        if train and np.random.rand() < self.epsilon:
            return torch.randint(0, self.n_actions, size=())
        
        with torch.no_grad():
            return (self.model(state)).argmax()
    
    def learning_step(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        batch_size = len(dones)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        not_dones = ~torch.tensor(dones, dtype=torch.bool)
  
        q_values = self.model(states)[range(batch_size), actions]

        with torch.no_grad():
            target_actions = self.model(next_states).argmax(-1)
            target_q_values = self.target_model(next_states)
            next_state_values = target_q_values[
                range(batch_size), target_actions]
            
        q_target = rewards + not_dones * self.gamma * next_state_values

        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, q_target)
        loss.backward()
        self.optimizer.step()

