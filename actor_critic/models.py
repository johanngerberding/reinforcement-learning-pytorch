import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvActorCritic(nn.Module):
    def __init__(self, observation_shape, action_space):
        super(ConvActorCritic, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.conv_output_size = self._get_conv_output_size(observation_shape)
        self.actor = nn.Linear(self.conv_output_size, action_space)
        self.critic = nn.Linear(self.conv_output_size, 1)


    def forward(self, x):
        x = x.float() / 256
        x = self.backbone(x)
        return self.actor(x), self.critic(x)


# shared backbone model
class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden):
        super(ActorCritic, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_space, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, hidden, bias=True),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_space, bias=True)
        self.critic = nn.Linear(hidden, 1, bias=True)


    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.backbone(x)
        return self.actor(x), self.critic(x)


class Critic(nn.Module):
    def __init__(self, observation_space, hidden_size):
        super(Critic, self).__init__()
        self.model= nn.Sequential(nn.Linear(observation_space, hidden_size, bias=True),
                nn.ReLU(), nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(), nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.model(x)


class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super(Actor, self).__init__()
        self.model= nn.Sequential(
                nn.Linear(observation_space, hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_size, action_space, bias=True),
        )

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.model(x)


