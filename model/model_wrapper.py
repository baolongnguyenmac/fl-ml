import torch
import torch.nn as nn
import flwr as fl
from flwr.common import Weights
from collections import OrderedDict
from learn2learn.algorithms.meta_sgd import MetaSGD, meta_sgd_update, clone_module, clone_parameters

import copy

FED_AVG = "FedAvg"
FED_META_MAML = "FedMetaMAML"
FED_AVG_META = "FedAvgMeta"
FED_META_SDG = "FedMetaSGD"

FEMNIST_MODEL = "femnist"
SHAKESPEARE_MODEL = "shakespeare"
SENT140_MODEL = "sent140"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelWrapper:
    """generate a wrapper that wraps a nn.Module
    """
    def __init__(self, model: nn.Module, model_name):
        self.model = model
        self.model_name = model_name
        self.model = self.model.to(DEVICE)

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

# class MetaSGDModelWrapper(nn.Module):
#     def __init__(self, model: nn.Module, lr: float = 0.01) -> None:
#         super().__init__()
#         self.model = model
#         lrs = [torch.ones_like(p) * lr for p in model.parameters()]
#         lrs = nn.ParameterList([nn.Parameter(lr) for lr in lrs])
#         self.alpha = lrs

#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)

#     def clone(self):
#         return copy.deepcopy(self)

#     def adapt(self, loss):
#         gradients = torch.autograd.grad(loss,
#                         self.model.parameters(),
#                         retain_graph=True,
#                         create_graph=True, allow_unused=True)
#         self.model = meta_sgd_update(self.model, self.alpha, gradients)
