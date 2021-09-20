import sys 
sys.path.insert(0, '../')
from model import femnist_model, shakespeare_model
from strategy_client.conventional_ml import ConventionalTest, ConventionalTrain

import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from typing import Tuple

FED_AVG = "FedAvg"
FED_META_MAML = "FedMetaMAML"
FED_AVG_META = "FedAvgMeta"
FED_META_SDG = "FedMetaSGD"

FEMNIST_MODEL = "femnist_model"
SHAKESPEARE_MODEL = "shakespeare_model"
SENT140_MODEL = "sent140_model"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model:
    """generate a model
    """
    def __init__(self, model: str, strategy: str):
        self.strategy = strategy
        self.model: nn.Module = None

        if model == FEMNIST_MODEL:
            self.model: nn.Module = femnist_model.Femnist()
        elif model == SHAKESPEARE_MODEL:
            self.model: nn.Module = shakespeare_model.Shakespeare()
        elif model == SENT140_MODEL:
            pass
        else:
            print("wrong model syntax")

        self.model = self.model.to(DEVICE)

    def load_model(self):
        """Load model

        Returns:
            nn.Module: an instance of class nn.Module
        """
        return self.model

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def train(
            self,
            trainloader: torch.utils.data.DataLoader, 
            epochs: int, 
            device: torch.device, 
        ) -> None:
        """train model at client

        Args:
            trainloader (torch.utils.data.DataLoader): train dataloader
            epochs (int): number of epochs
            device (torch.device): device
            strategy (str): fedAvg, fedAvgMeta, fedMetaMAML, defMetaSGD
        """
        trainer = None
        if self.strategy == FED_AVG:
            trainer = ConventionalTrain(
                self.model, 
                nn.functional.cross_entropy, 
                torch.optim.Adam(self.model.parameters(), lr=0.001), 
                device)
        elif self.strategy == FED_AVG_META:
            pass
        elif self.strategy == FED_META_MAML:
            pass
        elif self.strategy == FED_META_SDG:
            pass

        if trainer is not None:
            trainer.train(trainloader, epochs)
        else:
            print("wrong algorithm syntax")

    def test(
            self,
            testloader: torch.utils.data.DataLoader, 
            device: torch.device,
        ) -> Tuple[float, float]:
        """test the model at client

        Args:
            net (nn.Module): model
            testloader (torch.utils.data.DataLoader): test dataloader
            device (torch.device): device

        Returns:
            Tuple[float, float]: loss and accuracy
        """
        tester = ConventionalTest(self.model, nn.functional.cross_entropy, device)
        return tester.test(testloader)


''' test model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from data.dataloaders import femnist as dataloader
loader, size = dataloader.get_loader('../data/femnist/test/0/support.pickle')
model = Model(FEMNIST_MODEL, FED_AVG)
outs = model.test(loader, DEVICE)
print(outs) # a tuple that contains loss and acc
'''