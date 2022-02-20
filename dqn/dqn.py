import math
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, sigma, bias=True):
        super(NoisyLinearLayer, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
    
    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias 
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data 
        weight = self.weight 
        weight = weight + self.sigma_weight * self.epsilon_weight.data
        return F.linear(input, weight, bias) 


class NoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma, bias):
        raise NotImplementedError     


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
    

class NoisyDQN(nn.Module):
    def __init__(self, in_channels, in_shape, action_space):
        super(NoisyDQN, self).__init__()
        self.in_channels = in_channels
        self.action_space = action_space
        self.in_shape = in_shape

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out_shape(
            (self.in_channels, self.in_shape[0], self.in_shape[1]))
        self.flatten = nn.Flatten()
        self.noisy_layers = [
            NoisyLinearLayer(conv_out_size, 512, 0.017, bias=True),
            NoisyLinearLayer(512, self.action_space, 0.017, bias=True),
        ]
        self.noisy_fcs = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.noisy_fcs(x)
        return x
    
    def _get_conv_out_shape(self, inp_shape):
        out = self.conv(torch.zeros(1, *inp_shape))
        return int(np.prod(out.size()))
    
    def noisy_layers_sigma_snr(self):
        "Signal-to-Noise-Ratio"
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]