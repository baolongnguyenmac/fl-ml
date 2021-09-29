import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

import torch
import torch.nn as nn
import timeit

import sys
sys.path.insert(0, '../')
from model import model
from data.dataloaders import femnist as femnist_loader
from data.dataloaders import shakespeare as shakespeare_loader
from data.dataloaders import sent140 as sent140_loader
from strategy_client.conventional_ml import ConventionalTest
from strategy_client.fedmeta_sgd import MetaSGDTrain

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedMetaSGD(fl.client.Client):
    def __init__(self, cid: int, model: model.Model) -> None:
        super().__init__()
        self.model = model
        self.cid = cid

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def get_loader(self, train: bool, batch_size):
        tr = ''
        if train: tr = 'support'
        else: tr = 'query'

        if self.model.model_name == model.FEMNIST_MODEL:
            return femnist_loader.get_loader(f'./data/{self.model.model_name}/train/{self.cid}/{tr}.pickle', batch_size)
        elif self.model.model_name == model.SHAKESPEARE_MODEL:
            return shakespeare_loader.get_loader(f'./data/{self.model.model_name}/train/{self.cid}/{tr}.pickle', batch_size)
        elif self.model.model_name == model.SENT140_MODEL:
            return sent140_loader.get_loader(f'./data/{self.model.model_name}/train/{self.cid}/{tr}.pickle', batch_size)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()
        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        self.model.set_weights(weights)

        # get support_loader and query_loader
        support_loader, n_sample_support = self.get_loader(train=True, batch_size=batch_size)
        query_loader, _ = self.get_loader(train=False, batch_size=batch_size)

        trainer = MetaSGDTrain(
            self.model.model,
            nn.functional.cross_entropy,
            DEVICE
        )
        grads = trainer.train(support_loader, query_loader, epochs)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = grads
        params_prime = fl.common.weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=n_sample_support,
            num_examples_ceil=n_sample_support,
            fit_duration=fit_duration
        )
