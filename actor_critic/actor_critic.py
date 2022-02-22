import numpy as np 
import torch 
import torch.nn as nn 


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()
        # shared convolutional backbone
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        conv_out_size = self._get_conv_size(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
        
    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.flatten(self.conv(fx))
        return self.policy(conv_out), self.value(conv_out)
        
        
    def _get_conv_size(self, shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))