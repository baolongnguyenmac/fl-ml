import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, ParametersRes, Weights, weights_to_parameters, parameters_to_weights
import torch
from typing import Dict
from flwr.common.typing import Scalar

import sys
sys.path.insert(0, '../')
from client_trainer.conventional_trainer_tester import ConventionalTester
from model.model_wrapper import ModelWrapper, FEMNIST_MODEL, SHAKESPEARE_MODEL, SENT140_MODEL
from data.dataloaders.femnist import get_loader as femnist_loader
from data.dataloaders.sent140 import get_loader as sent140_loader
from data.dataloaders.shakespeare import get_loader as shakespeare_loader


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseClient(fl.client.Client):
    def __init__(self, model_wrapper: ModelWrapper, cid: str) -> None:
        super().__init__()
        self.model_wrapper = model_wrapper
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

    def get_parameters(self) -> ParametersRes:
        print(f"[Client {self.cid}]: get_parameters")

        weights: Weights = self.model_wrapper.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_loader(self, support: bool, batch_size: int = 32):
        s = 'support' if support else 'query'

        if self.model_wrapper.model_name == FEMNIST_MODEL:
            return femnist_loader(f'./data/{self.model_wrapper.model_name}/train/{self.cid}/{s}.pickle', batch_size)
        elif self.model_wrapper.model_name == SHAKESPEARE_MODEL:
            return shakespeare_loader(f'./data/{self.model_wrapper.model_name}/train/{self.cid}/{s}.pickle', batch_size)
        elif self.model_wrapper.model_name == SENT140_MODEL:
            return sent140_loader(f'./data/{self.model_wrapper.model_name}/train/{self.cid}/{s}.pickle', batch_size)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}]: Evaluate in round {ins.config['current_round']}")
        weights = parameters_to_weights(ins.parameters)
        config = ins.config
        batch_size = int(config["batch_size"])

        self.model_wrapper.set_weights(weights)

        query_loader, num_query = self.get_loader(False, batch_size)
        tester = ConventionalTester(
            self.model_wrapper,
            torch.nn.functional.cross_entropy if self.model_wrapper.model_name != 'sent140' else torch.nn.functional.binary_cross_entropy,
            DEVICE
        )

        loss, acc = tester.test(query_loader)
        acc = acc/num_query
        print(f'[Client {self.cid}]: Evaluate loss: {loss}, Evaluate accuracy: {acc}')

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=loss, num_examples=num_query, metrics={'acc': acc}
        )
