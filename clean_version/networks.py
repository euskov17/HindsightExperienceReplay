import numpy as np
import torch
import torch.nn as nn

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)



class CriticNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size=128):
        super().__init__()

        self.model = MLPLayer(state_dim + actions_dim, 1, hidden_size)

    def forward(self, states, actions):
        sa = torch.cat([states, actions], dim=-1)
        return self.model(sa).squeeze()


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size=128, max_action=1.0):
        super().__init__()
        
        self.model = MLPLayer(state_dim, actions_dim, hidden_size)

        self.max_action = max_action

    def forward(self, input):
        return torch.tanh(self.model(input)) * self.max_action