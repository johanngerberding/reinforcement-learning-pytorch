import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
            self,
            observation_space: int,
            action_space: int,
            hidden_dim: int,
            gamma: float = 0.99
    ):
        super(DQN, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        
        self.fc1 = nn.Linear(observation_space, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_space)
        
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return self.fc4(x)


class DQN_Conv(nn.Module):
    def __init__(self, in_channels, action_space):
        super(DQN_Conv, self).__init__()
        self.in_channels = in_channels
        self.action_space = action_space

        self.conv1 = nn.Conv2d(
            self.in_channels, 32,
            kernel_size=8, stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(self.flatten(x)))
        return self.fc5(x)