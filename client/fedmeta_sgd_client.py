import torch
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, Weights, weights_to_parameters, parameters_to_weights
import pickle

from .base_client import BaseClient
from client_worker.meta_sgd_worker import MetaSGDTrainer, MetaSGDTester
from model.model_wrapper import ModelWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedMetaSGDClient(BaseClient):
    def __init__(self, model_wrapper: ModelWrapper, cid: str, mode: str, num_eval_clients: int, per_layer=None) -> None:
        super().__init__(model_wrapper, cid, mode, num_eval_clients)
        self.per_layer = (-1) * per_layer * 2 if per_layer is not None else None

    def fit(self, ins: FitIns) -> FitRes:
        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config

        # Get training config
        current_round = int(config["current_round"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        beta = float(config["beta"])

        # set weight of server to client
        self.model_wrapper.set_weights(weights)

        # assgin personalized layer to local model 
        if self.per_layer is not None:
            print('running fedPer 1', self.per_layer)
            try:
                with open(f'./personalized_weight/{self.cid}.pickle', 'rb') as input:
                    personalized_weight = pickle.load(input)
                weights[self.per_layer:] = personalized_weight
            except:
                pass

        # train the model
        trainer = MetaSGDTrainer(
            model_wrapper=self.model_wrapper,
            device=DEVICE,
            cid=self.cid,
            current_round=current_round,
            batch_size=batch_size,
            epochs=epochs,
            beta=beta
        )
        training_loss, training_acc, num_training_sample = trainer.train()

        # return the refined weights and the number of examples used for training
        new_weights: Weights = self.model_wrapper.get_weights()
        new_params = weights_to_parameters(new_weights)

        # save personalized layer to file
        if self.per_layer is not None:
            print('running fedPer 2', self.per_layer)
            personalized_weight = new_weights[self.per_layer:]
            with open(f'./personalized_weight/{self.cid}.pickle', 'wb') as fp:
                pickle.dump(personalized_weight, fp)
            fp.close()

        return FitRes(
            parameters=new_params,
            num_examples=num_training_sample,
            metrics={'training_loss': training_loss, 'training_accuracy': training_acc}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        weights = parameters_to_weights(ins.parameters)
        config = ins.config
        current_round = int(config["current_round"])
        batch_size = int(config["batch_size"])
        epochs = int(config['epochs'])

        self.model_wrapper.set_weights(weights)

        # test the model
        tester = MetaSGDTester(
            model_wrapper=self.model_wrapper,
            device=DEVICE,
            cid=self.cid,
            current_round=current_round,
            batch_size=batch_size,
            num_eval_clients=self.num_eval_clients,
            mode=self.mode,
            epochs=epochs
        )
        val_loss, val_acc, num_val_sample = tester.test()

        return EvaluateRes(
            loss=val_loss, num_examples=num_val_sample, metrics={'acc': val_acc}
        )
