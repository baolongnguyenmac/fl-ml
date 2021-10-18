import torch
import torch.nn as nn
from flwr.common import FitIns, FitRes, Weights, weights_to_parameters, parameters_to_weights

import sys 
sys.path.insert(0, '../')
from client.base_client import BaseClient
from client_trainer.meta_sgd_trainer import MetaSGDTrainer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedMetaSGDClient(BaseClient):
    def fit(self, ins: FitIns) -> FitRes:
        print(f'[Client {self.cid}]: Fit')

        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        beta = float(config["beta"])

        # set weight of server to client
        self.model_wrapper.set_weights(weights)

        # get loader
        support_loader, num_examples_support = self.get_loader(
            True, batch_size)
        query_loader, num_examples_query = self.get_loader(False, batch_size)

        trainer = MetaSGDTrainer(
            self.model_wrapper,
            nn.functional.cross_entropy if self.model_wrapper.model_name != 'sent140' else nn.functional.binary_cross_entropy,
            DEVICE,
            self.cid,
            torch.optim.Adam(self.model_wrapper.model.parameters(), lr=beta)
        )
        print(f'[Client {self.cid}] Fit {epochs} epoch(s) on {len(support_loader)} batch(es) using {DEVICE}')
        training_loss, training_acc = trainer.train(support_loader, query_loader, epochs, beta)
        training_acc /= num_examples_query
        print(f'[Client {self.cid}]: Training_loss: {training_loss}, Training acc: {training_acc}')

        # return the refined weights and the number of examples used for training
        new_weights: Weights = self.model_wrapper.get_weights()
        new_params = weights_to_parameters(new_weights)
        return FitRes(
            parameters=new_params,
            num_examples=num_examples_query,
            metrics={'training_loss': training_loss, 'training_accuracy': training_acc}
        )
