from flwr.common import FitIns, FitRes, Weights, weights_to_parameters, parameters_to_weights

import torch
import torch.nn as nn
import timeit

import sys
sys.path.insert(0, '../')
from client.base_client import BaseClient
from model import model
from strategy_client.fedmeta_maml import MAMLTrain

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedMetaMAMLClient(BaseClient):
    def __init__(self, cid: int, model: model.Model) -> None:
        super().__init__(cid=cid, model=model)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()
        # Get training config
        alpha = float(config["alpha"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        self.model.set_weights(weights)

        # Train model
        supportloader, num_examples_train = self.get_loader(train=True, batch_size=batch_size)

        trainer = MAMLTrain(
            self.model.model,
            nn.functional.cross_entropy,    
            torch.optim.Adam(self.model.model.parameters(), alpha),
            DEVICE
        )
        trainer.trainOnSupport(supportloader, epochs)
        
        queryloader, _ = self.get_loader(train=False, batch_size=batch_size)

        grad = trainer.trainOnQuey(queryloader)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = grad
        params_prime = weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )
