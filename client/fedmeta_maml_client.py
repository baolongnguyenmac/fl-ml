import torch
import torch.nn as nn
from flwr.common import FitIns, FitRes, Weights, weights_to_parameters, parameters_to_weights

import sys
sys.path.insert(0, '../')
from client.base_client import BaseClient
from client_trainer.maml_trainer import MAMLTrainer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FedMetaMAMLClient(BaseClient):
    def fit(self, ins: FitIns) -> FitRes:
        print(f'[Client {self.cid}]: Fit')

        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config

        # Get training config
        alpha = float(config["alpha"])
        beta = float(config['beta'])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        self.model_wrapper.set_weights(weights)

        # get loader
        support_loader, num_examples_train = self.get_loader(
            True, batch_size=batch_size)
        query_loader, _ = self.get_loader(False, batch_size=batch_size)

        trainer = MAMLTrainer(
            self.model_wrapper,
            nn.functional.cross_entropy if self.model_wrapper.model_name != 'sent140' else nn.functional.binary_cross_entropy,
            torch.optim.Adam(self.model_wrapper.model.parameters(), alpha),
            torch.optim.Adam(self.model_wrapper.model.parameters(), beta),
            DEVICE,
            self.cid
        )
        trainer.train(support_loader, query_loader, epochs)

        # Return the refined weights and the number of examples used for training
        new_weights: Weights = self.model_wrapper.get_weights()
        new_params = weights_to_parameters(new_weights)
        return FitRes(
            parameters=new_params,
            num_examples=num_examples_train,
        )
