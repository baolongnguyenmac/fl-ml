import torch
import flwr as fl 
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights, weights_to_parameters, parameters_to_weights
import timeit

import sys
sys.path.insert(0, '../')
from strategy_client.conventional_ml import ConventionalTest
from strategy_client.fedmeta_sgd import MetaSGDTrain
from model import model
from data.dataloaders import (
    femnist as femnist_loader,
    shakespeare as shakespeare_loader,
    sent140 as sent140_loader
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedMetaSGDClient(fl.client.Client):
    def __init__(self, cid: int, model: model.Model) -> None:
        super().__init__()
        print(f'Client {cid}: Init')
        self.model = model
        self.cid = cid

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def get_loader(self, train: bool, batch_size):
        tr = 'support' if train else 'query'

        if self.model.model_name == model.FEMNIST_MODEL:
            return femnist_loader.get_loader(f'./data/{self.model.model_name}/train/{self.cid}/{tr}.pickle', batch_size)
        elif self.model.model_name == model.SHAKESPEARE_MODEL:
            return shakespeare_loader.get_loader(f'./data/{self.model.model_name}/train/{self.cid}/{tr}.pickle', batch_size)
        elif self.model.model_name == model.SENT140_MODEL:
            return sent140_loader.get_loader(f'./data/{self.model.model_name}/train/{self.cid}/{tr}.pickle', batch_size)

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

        trainer = MetaSGDTrain(
            self.model.model, 
            torch.nn.functional.cross_entropy, 
            DEVICE)

        grads = trainer.train(support_loader, query_loader, epochs)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = grads
        params_prime = weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_sample_support,
            num_examples_ceil=num_sample_support,
            fit_duration=fit_duration
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = parameters_to_weights(ins.parameters)
        config = ins.config
        batch_size = int(config["batch_size"])

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # get data
        test_loader, num_sample_test  = self.get_loader(train=False, batch_size=batch_size)

        tester = ConventionalTest(self.model.model, torch.nn.functional.cross_entropy, DEVICE)
        loss, acc = tester.test(test_loader)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=float(loss), num_examples=num_sample_test, accuracy=float(acc)
        )