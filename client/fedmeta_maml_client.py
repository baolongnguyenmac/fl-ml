import torch
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, Weights, weights_to_parameters, parameters_to_weights

from .base_client import BaseClient
from client_worker.maml_worker import MAMLTester, MAMLTrainer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedMetaMAMLClient(BaseClient):
    def fit(self, ins: FitIns) -> FitRes:
        # Get training config
        config = ins.config
        current_round = int(config['current_round'])
        alpha = float(config["alpha"])
        beta = float(config['beta'])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        weights: Weights = parameters_to_weights(ins.parameters)
        self.model_wrapper.set_weights(weights)

        # train the model
        trainer = MAMLTrainer(
            model_wrapper=self.model_wrapper,
            device=DEVICE,
            cid=self.cid,
            current_round=current_round,
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
            beta=beta
        )
        training_loss, training_acc, num_train_sample = trainer.train()

        # Return the refined weights and the number of examples used for training
        new_weights: Weights = self.model_wrapper.get_weights()
        new_params = weights_to_parameters(new_weights)
        return FitRes(
            parameters=new_params,
            num_examples=num_train_sample,
            metrics={'training_loss': training_loss, 'training_accuracy': training_acc}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config

        # Get training config
        current_round = int(config['current_round'])
        alpha = float(config["alpha"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        self.model_wrapper.set_weights(weights)

        # test the model
        tester = MAMLTester(
            model_wrapper=self.model_wrapper,
            device=DEVICE,
            cid=self.cid,
            current_round=current_round,
            batch_size=batch_size,
            num_eval_clients=self.num_eval_clients,
            mode=self.mode,
            epochs=epochs,
            alpha=alpha
        )
        val_loss, val_acc, num_val_sample = tester.test()

        return EvaluateRes(
            loss=val_loss, num_examples=num_val_sample, metrics={'acc': val_acc}
        )
