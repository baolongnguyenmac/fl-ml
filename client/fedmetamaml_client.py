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
from strategy_client.fedmeta_maml import MAMLTrain
from strategy_client.conventional_ml import ConventionalTest

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedMetaMAMLClient(fl.client.Client):
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
        tr = 'support' if train else 'query'

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
        config = ins.config
        batch_size = int(config["batch_size"])

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader, num_examples = self.get_loader(train=False, batch_size=batch_size)
        tester = ConventionalTest(
            self.model.model,
            nn.functional.cross_entropy,
            DEVICE
        )
        loss, accuracy = tester.test(testloader)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=float(loss), num_examples=num_examples, accuracy=float(accuracy)
        )
