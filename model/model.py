import sys 
sys.path.insert(0, '../')
from model import femnist_model, shakespeare_model

import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from typing import Tuple

FED_AVG = "FedAvg"
FED_META_MAML = "FedMetaMAML"
FED_AVG_META = "FedAvgMeta"
FED_META_SDG = "FedMetaSGD"

FEMNIST_MODEL = "femnist"
SHAKESPEARE_MODEL = "shakespeare"
SENT140_MODEL = "sent140"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model:
    """generate a model
    """
    def __init__(self, model: str, strategy: str):
        self.strategy = strategy
        self.model: nn.Module = None
        self.model_name = model

        if model == FEMNIST_MODEL:
            self.model: nn.Module = femnist_model.Femnist()
        elif model == SHAKESPEARE_MODEL:
            self.model: nn.Module = shakespeare_model.Shakespeare()
        elif model == SENT140_MODEL:
            self.model: nn.Module = sent140_model.Sent140()
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
