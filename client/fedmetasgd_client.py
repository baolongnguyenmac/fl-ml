import torch
from flwr.common import FitIns, FitRes, Weights, weights_to_parameters, parameters_to_weights
import timeit
import torch.nn as nn

import sys
sys.path.insert(0, '../')

from model import model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        beta = float(config["beta"])

        # Set model parameters
        self.model.set_weights(weights)

        # data for training
        support_loader, num_sample_support = self.get_loader(train=True, batch_size=batch_size)
        query_loader, _ = self.get_loader(train=False, batch_size=batch_size)
        if self.model.model_name=='sent140':
            trainer = MetaSGDTrain(
                self.model.model, 
                torch.nn.functional.binary_cross_entropy, 
                DEVICE,
                self.cid)
        else:
            trainer = MetaSGDTrain(
                self.model.model, 
                torch.nn.functional.cross_entropy, 
                DEVICE,
                self.cid)

        trainer.train(support_loader, query_loader, epochs, beta)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        params_prime = weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_sample_support,
            num_examples_ceil=num_sample_support,
            fit_duration=fit_duration
        )
