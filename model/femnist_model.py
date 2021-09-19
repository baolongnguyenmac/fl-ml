from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import flwr as fl

import sys 
sys.path.append('../')
from strategy_client.conventional_ml import ConventionalTrain, ConventionalTest

DATA_ROOT = "../data/femnist"
FED_AVG = "FedAvg"
FED_META_MAML = "FedMetaMAML"
FED_AVG_META = "FedAvgMeta"
FED_META_SDG = "FedMetaSGD"
# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


def load_model():
    """Load Femnist model

    Returns:
        Femnist: an instance of class Femnist
    """
    return Femnist()


def train(
        net: nn.Module, 
        trainloader: torch.utils.data.DataLoader, 
        epochs: int, 
        device: torch.device, 
        strategy: str = FED_AVG
    ) -> None:
    """train model at client

    Args:
        net (nn.Module): model
        trainloader (torch.utils.data.DataLoader): train dataloader
        epochs (int): number of epochs
        device (torch.device): device
        strategy (str): fedAvg, fedAvgMeta, fedMetaMAML, defMetaSGD
    """
    trainer = None
    if strategy == FED_AVG:
        trainer = ConventionalTrain(
            net, 
            nn.CrossEntropyLoss(), 
            torch.optim.Adam(net.parameters(), lr=0.001), 
            device)
    elif strategy == FED_AVG_META:
        pass
    elif strategy == FED_META_MAML:
        pass
    elif strategy == FED_META_SDG:
        pass

    if trainer is not None:
        trainer.train(trainloader, epochs)
    else:
        print("wrong algorithm syntax")


def test(
        net: nn.Module, 
        testloader: torch.utils.data.DataLoader, 
        device: torch.device,
        # strategy: str
    ) -> Tuple[float, float]:
    """test the model at client

    Args:
        net (nn.Module): model
        testloader (torch.utils.data.DataLoader): test dataloader
        device (torch.device): device

    Returns:
        Tuple[float, float]: loss and accuracy
    """
    tester = ConventionalTest(net, nn.CrossEntropyLoss(), device)
    return tester.test(testloader)

'''
    # test
    from data.dataloaders import femnist as dataloader
    loader, size = dataloader.get_loader('../data/femnist/test/0/support.pickle')
    for x, y in loader:
        # x = x.reshape(-1, 1, 28, 28)
        # print(x.shape) # [32, 1, 28, 28]
        print(x.shape) # [32, 28*28]
        print(y.shape) # [32]
        model = load_model()
        outs = model(x)
        print(outs.shape) # [32, 62]
        break
'''
