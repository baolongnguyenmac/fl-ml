import torch
import torch.nn as nn
import flwr as fl
from flwr.common import Weights
from collections import OrderedDict
from learn2learn.algorithms.meta_sgd import MetaSGD, meta_sgd_update, clone_module, clone_parameters

FED_AVG = "FedAvg"
FED_META_MAML = "FedMetaMAML"
FED_AVG_META = "FedAvgMeta"
FED_META_SGD = "FedMetaSGD"

FEMNIST_MODEL = "femnist"
MNIST_MODEL = "mnist"
SENT140_MODEL = "sent140"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelWrapper:
    """generate a wrapper that wraps a nn.Module
    """
    def __init__(self, model: nn.Module, model_name: str):
        self.model = model.to(DEVICE)
        self.model_name = model_name

    def get_weights(self) -> Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict, strict=True)

class MetaSGDModelWrapper(MetaSGD):
    """generate a meta model that wraps a nn.Module
    """
    def clone(self):
        return MetaSGDModelWrapper(clone_module(self.module),
                        lrs=clone_parameters(self.lrs),
                        first_order=self.first_order)

    def adapt(self, loss, first_order=None):
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = torch.autograd.grad(loss,
                        self.module.parameters(),
                        retain_graph=second_order,
                        create_graph=second_order, allow_unused=True)
        self.module = meta_sgd_update(self.module, self.lrs, gradients)
