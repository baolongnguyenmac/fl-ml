import torch
from flwr.common import FitIns, FitRes, Weights, weights_to_parameters, parameters_to_weights
import timeit
import torch.nn as nn

import sys
sys.path.insert(0, '../')

from model import model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetaSGD(nn.Module):
    def __init__(self, model: nn.Module, lr=0.01):
        super().__init__()
        self.model = model
        alpha = [torch.ones_like(p) * lr for p in self.model.parameters()]
        alpha = nn.ParameterList([nn.Parameter(lr) for lr in alpha])
        self.alpha = alpha
        self.count = 0

    def forward(self, x):
        return self.model(x)

    def _adapt_update(self, model: nn.Module, grads):
        # Update the params
        if len(list(model._modules)) == 0 and len(list(model._parameters)) != 0:
            for param_key in model._parameters:
                p = model._parameters[param_key].detach().clone()
                try:
                    model._parameters[param_key] = p - self.alpha[self.count] * grads[self.count]
                except:
                    pass
                self.count += 1

        # Then, recurse for each submodule
        for module_key in model._modules:
            model._modules[module_key] = self._adapt_update(model._modules[module_key], grads)
        return model

    def adapt(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        self._adapt_update(self.model, grads)

from strategy_client.fedmeta_sgd import MetaSGDTrain        
from client.base_client import BaseClient

class FedMetaSGDClient(BaseClient):
    def __init__(self, cid: int, model: model.Model) -> None:
        super().__init__(cid=cid, model=model)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = parameters_to_weights(ins.parameters)
        # print('\n\n', self.cid, weights, '\n\n')
        config = ins.config
        fit_begin = timeit.default_timer()
        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        self.model.set_weights(weights)

        # Set model parameters
        self.model.set_weights(weights)

        # data for training
        support_loader, num_sample_support = self.get_loader(train=True, batch_size=batch_size)
        query_loader, _ = self.get_loader(train=False, batch_size=batch_size)
        if self.model.model_name=='sent140':
            trainer = MetaSGDTrain(
                self.model.model, 
                torch.nn.functional.binary_cross_entropy, 
                DEVICE)
        else:
            trainer = MetaSGDTrain(
                self.model.model, 
                torch.nn.functional.cross_entropy, 
                DEVICE)

        grads = trainer.train(support_loader, query_loader, epochs)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = grads
        print('\n\n', grads, '\n\n')
        params_prime = weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_sample_support,
            num_examples_ceil=num_sample_support,
            fit_duration=fit_duration
        )
