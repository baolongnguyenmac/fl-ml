from flwr.common import FitIns, FitRes, Weights, weights_to_parameters, parameters_to_weights
import torch

import sys
sys.path.insert(0, '../')
from client_trainer.conventional_trainer_tester import ConventionalTrainer
from client.base_client import BaseClient


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FedAvgClient(BaseClient):
    def fit(self, ins: FitIns) -> FitRes:
        print(f'[Client {self.cid}]: Fit')

        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config

        # get training config
        epochs = int(config['epochs'])
        batch_size = int(config['batch_size'])
        lr = float(config['alpha'])

        # set weight of server to client
        self.model_wrapper.set_weights(weights)

        # get loader
        support_loader, num_support = self.get_loader(True, batch_size)

        # train model
        trainer = ConventionalTrainer(
            self.model_wrapper.model,
            torch.nn.functional.cross_entropy if self.model_wrapper.model_name != 'sent140' else torch.nn.functional.binary_cross_entropy,
            torch.optim.Adam(self.model_wrapper.model.parameters(), lr=lr),
            DEVICE,
            self.cid
        )
        trainer.train(support_loader, epochs)

        # return the refined weights and the number of examples used for training
        new_weights: Weights = self.model_wrapper.get_weights()
        new_params = weights_to_parameters(new_weights)
        return FitRes(
            parameters=new_params,
            num_examples=num_support
        )
