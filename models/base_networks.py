import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size, 
                    hidden_size=128, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        
        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

class Critic(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size=128, num_layers=3):
        super().__init__()
        self.model = MLPLayer(state_dim + actions_dim, 1, hidden_size, num_layers)

    def forward(self, states, actions):
        sa = torch.cat([states, actions], -1)
        return self.model(sa).squeeze()
