import torch
import torch.nn as nn

import sys 
sys.path.append('../')
from strategy_client.conventional_ml import ConventionalTrain, ConventionalTest

class Femnist(nn.Module):
    def __init__(self) -> None:
        super(Femnist, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2), # batch: [#numOfElement, 64, 14, 14]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2), # batch: [#numOfElement, 64, 7, 7]
            
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64*7*7, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=62),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, 28, 28)
        return self.network(x)
