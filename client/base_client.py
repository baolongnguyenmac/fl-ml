import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, ParametersRes, Weights, weights_to_parameters, parameters_to_weights
import torch
from typing import Dict
from flwr.common.typing import Scalar
import random 

import sys
sys.path.insert(0, '../')
from client_trainer.conventional_trainer_tester import ConventionalTester, ConventionalTrainer
from model.model_wrapper import ModelWrapper, FED_AVG, FED_META_MAML, FED_AVG_META, FED_META_SDG, FEMNIST_MODEL, SHAKESPEARE_MODEL, SENT140_MODEL
from data.dataloaders.femnist import get_loader as femnist_loader
from data.dataloaders.sent140 import get_loader as sent140_loader
from data.dataloaders.shakespeare import get_loader as shakespeare_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseClient(fl.client.Client):
    def __init__(self, model_wrapper: ModelWrapper, cid: str, mode: str, strategy: str, num_test_clients: int, num_val_client: int) -> None:
        super().__init__()
        self.model_wrapper = model_wrapper
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        # mode bao gồm 'train' và 'test', sử dụng mode để đổi tập valid/test trong lúc evaluate của client
        self.mode = mode 
        self.strategy = strategy
        self.num_test_clients = num_test_clients
        self.num_val_clients = num_val_client

    def get_parameters(self) -> ParametersRes:
        print(f"[Client {self.cid}]: get_parameters")

        weights: Weights = self.model_wrapper.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_loader(self, support: bool, batch_size: int = 32, mode: int = 0):
        if mode == 0:
            t = 'train'
            id_set = self.cid
        elif mode == 1:
            t = 'val'
            id_set = random.choice(list(range(self.num_val_clients)))
        elif mode == 2:
            t = 'test'
            id_set = random.choice(list(range(self.num_test_clients)))

        # t = 'train' if mode==0 else 'valid' if mode==1 else 'test'
        s = 'support' if support else 'query'

        if self.model_wrapper.model_name == FEMNIST_MODEL:
            loader = femnist_loader
        elif self.model_wrapper.model_name == SHAKESPEARE_MODEL:
            loader = shakespeare_loader
        elif self.model_wrapper.model_name == SENT140_MODEL:
            loader = sent140_loader

        return loader(f'./data/{self.model_wrapper.model_name}/{t}/{id_set}/{s}.pickle', batch_size)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # parse instruction (receive from server)
        weights = parameters_to_weights(ins.parameters)
        config = ins.config
        batch_size = int(config["batch_size"])
        epochs = int(config['epochs'])
        alpha = float(config['alpha'])

        # set mode to get loader
        if self.mode == 'train':
            mode = 1 # evaluate (in training phase) will use the valid set
        elif self.mode == 'test':
            mode = 2 # evaluate (in test phase) will use the test set

        # get loader
        support_loader, num_support = self.get_loader(True, batch_size, mode)
        query_loader, num_query = self.get_loader(False, batch_size, mode)

        # set weight
        self.model_wrapper.set_weights(weights)

        if self.strategy == FED_AVG:
            print(f"[Client {self.cid}]: Evaluate in round {ins.config['current_round']}")

            tester = ConventionalTester(
                self.model_wrapper,
                torch.nn.functional.cross_entropy if self.model_wrapper.model_name != 'sent140' else torch.nn.functional.binary_cross_entropy,
                DEVICE
            )

            loss, acc = tester.test(query_loader)
            acc = acc/num_query
            print(f'[Client {self.cid}]: Evaluate loss: {loss}, Evaluate accuracy: {acc}')

        elif self.strategy == FED_AVG_META or self.strategy == FED_META_MAML or self.strategy == FED_META_SDG:
            print(f"[Client {self.cid}]: Meta-Evaluate in round {ins.config['current_round']}")

            trainer = ConventionalTrainer(
                self.model_wrapper, 
                torch.nn.functional.cross_entropy if self.model_wrapper.model_name != 'sent140' else torch.nn.functional.binary_cross_entropy,
                torch.optim.Adam(self.model_wrapper.model.parameters(), lr=alpha),
                DEVICE
            )
            trainer.train(support_loader, epochs)

            tester = ConventionalTester(
                self.model_wrapper, 
                torch.nn.functional.cross_entropy if self.model_wrapper.model_name != 'sent140' else torch.nn.functional.binary_cross_entropy,
                DEVICE
            )
            loss, acc = tester.test(support_loader)
            acc = acc/num_query
            print(f'[Client {self.cid}]: Meta-Evaluate loss: {loss}, Meta-Evaluate accuracy: {acc}')

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=loss, num_examples=num_query, metrics={'acc': acc}
        )
