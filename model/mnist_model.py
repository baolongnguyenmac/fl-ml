import torch
import torch.nn as nn
import torch.nn.functional as F

class Mnist(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super().__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
