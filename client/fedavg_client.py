import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

import torch
import torch.nn as nn
import timeit

import sys
sys.path.insert(0, '../')
from model import model
from data.dataloaders import femnist as dataloader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedAvgClient(fl.client.Client):
    def __init__(self, cid: int, model: model.Model) -> None:
        super().__init__()
        self.model = model
        self.cid = cid

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

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

        # Train model
        trainloader, num_examples_train = dataloader.get_loader(f'../data/femnist/train/{self.cid}/support.pickle', batch_size)
        self.model.train(self.model, trainloader, epochs, DEVICE, model.FED_AVG)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader, num_examples = dataloader.get_loader(f'../data/femnist/train/{self.cid}/query.pickle')
        loss, accuracy = self.model.test(self.model, testloader, DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=float(loss), num_examples=num_examples, accuracy=float(accuracy)
        )