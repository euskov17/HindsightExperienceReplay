import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLP_layer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
    
    def forward(self, input):
        return self.model(input)
    
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.model = MLP_layer(input_dim, output_dim + 1, hidden_size)

    def forward(self, input):
        out = self.model(input)
        value = out[..., -1]
        advantage = out[..., :-1]
        advantage = advantage - advantage.mean(-1, keepdim=True)
        return (value[..., None] + advantage)

class DQN:
    def __init__(self, state_dim, n_actions, hidden_size=128, gamma=0.99):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma

        self.model = DuelingQNetwork(state_dim, n_actions, hidden_size)
        self.target_model = DuelingQNetwork(state_dim, n_actions, hidden_size)
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.epsilon = 0.2
        self.update_freq = 16
        self.update_iter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            return (self.model(state)).argmax().numpy().item()
    
    def learning_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        batch_size = len(dones)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
  
        q_values = self.model(states)[range(batch_size), actions]

        with torch.no_grad():
            target_q_values = self.target_model(next_states)
            next_state_values = torch.max(target_q_values, -1).values

        q_target = rewards + (1 - dones) * self.gamma * next_state_values

        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, q_target)
        loss.backward()
        self.optimizer.step()

        self.update_iter += 1
        if (self.update_iter % self.update_freq == 0):
            self.epsilon *= 0.99
            self.update_target_model()