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
# from strategy_client.conventional_ml import ConventionalTest, ConventionalTrain

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