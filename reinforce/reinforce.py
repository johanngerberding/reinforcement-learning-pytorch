import torch
import torch.nn as nn
import torch.nn.functional as F


class Pi(nn.Module):
    def __init__(
            self,
            observation_space: int,
            action_space: int,
            hidden_dim: int = 128,
            gamma: float = 0.99,
    ):
        super(Pi, self).__init__()
        self.data = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(self.observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_space)
        ]
        self.model = nn.Sequential(*layers)
        self.gamma = gamma

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=0)

    def put_data(self, item):
        self.data.append(item)

    def train(self, optimizer, device):
        R = 0.0
        optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob).to(device) * R
            loss.backward()

        optimizer.step()
        self.data = []
        return loss.item()
