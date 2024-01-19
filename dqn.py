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
        value = out[..., 0]
        advantage = out[..., 1:]
        advantage = advantage - advantage.mean(-1, keepdim=True)
        return value[..., None] + advantage

class DQN:
    def __init__(self, state_dim, n_actions, hidden_size=128, gamma=0.98):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.tau=0.005

        self.model = DuelingQNetwork(state_dim, n_actions, hidden_size)
        self.target_model = DuelingQNetwork(state_dim, n_actions, hidden_size)
        self.update_network_parameters()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.epsilon = 0.2

    def update_network_parameters(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
        # for q_target_params, q_eval_params in zip(self.target_model.parameters(), self.model.parameters()):
        #     q_target_params.data.copy_(self.tau * q_eval_params + (1 - self.tau) * q_target_params)

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
            target_actions = self.model(next_states).argmax(-1)
            target_q_values = self.target_model(next_states)
            next_state_values = target_q_values[range(batch_size), target_actions]
            
        q_target = rewards + (1 - dones) * self.gamma * next_state_values

        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, q_target)
        loss.backward()
        self.optimizer.step()


class DQNAgent:
    def __init__(self, state_size, num_actions, hidden_size=256):

        self.state_size = state_size
        self.num_actions = num_actions

        self.gamma = 0.98
        self.batch_size = 128
        self.train_start = 1000

        self.Q_network = DuelingQNetwork(state_size, num_actions, hidden_size=hidden_size)
        self.target_network = DuelingQNetwork(state_size, num_actions, hidden_size=hidden_size)
        self.update_network_parameters()
        self.epsilon=0.2
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=0.001)

    def update_network_parameters(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            return self.greedy_action(state)
        else:
            return torch.randint(self.num_actions, size=())

    def greedy_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            return self.Q_network(state).argmax()

    def learning_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        batch_size = len(dones)

        # state_batch = torch.tensor(states, dtype=torch.float)
        # action_batch = torch.tensor(actions, dtype=torch.int)
        # reward_batch = torch.tensor(rewards, dtype=torch.float)
        # next_states = torch.tensor(next_states, dtype=torch.float)
        # dones = torch.tensor(dones, dtype=bool)
  
        state_batch = torch.tensor(states, dtype=torch.float)
        action_batch = torch.tensor(actions, dtype=torch.int)
        reward_batch = torch.tensor(rewards, dtype=torch.float)
        non_final_mask = ~torch.tensor(dones, dtype=bool)
        next_states = torch.tensor(next_states, dtype=torch.float)
        non_final_next_states = next_states[non_final_mask]

        Q_values = self.Q_network(state_batch)[range(batch_size), action_batch]

        # Double DQN target #
        next_state_values = torch.zeros(batch_size)
        number_of_non_final = sum(non_final_mask)
        with torch.no_grad():
            argmax_actions = self.Q_network(non_final_next_states).argmax(1)
            next_state_values[non_final_mask] = self.target_network(non_final_next_states)[
                range(number_of_non_final), argmax_actions]

        Q_targets = reward_batch + self.gamma * next_state_values
        #####################

        assert Q_values.shape == Q_targets.shape

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_values, Q_targets)
        loss.backward()
        self.optimizer.step()