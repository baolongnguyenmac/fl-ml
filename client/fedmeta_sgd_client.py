import torch
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, Weights, weights_to_parameters, parameters_to_weights

from .base_client import BaseClient
from client_worker.meta_sgd_worker import MetaSGDWorker
from model.model_wrapper import ModelWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedMetaSGDClient(BaseClient):
    def __init__(self, model_wrapper: ModelWrapper, cid: str, per_layer: int = None, new_client: bool = None) -> None:
        super().__init__(model_wrapper, cid, per_layer, new_client)
        self.worker = MetaSGDWorker(device=DEVICE, cid=self.cid, new_client=self.new_client)

    def fit(self, ins: FitIns) -> FitRes:
        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config

        # Get training config
        current_round = int(config["current_round"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        beta = float(config["beta"])      

        # assgin personalized layer to local model if the self.per_layer is specified
        if self.per_layer:
            # self.load_personalization_weight(self.cid, weights)
            self.model_wrapper.load_personalization_weight(self.cid, weights, self.per_layer, meta_sgd=True)
        else:
            self.model_wrapper.set_weights(weights)

        # train the model
        training_loss, training_acc, num_training_sample = self.worker.train(
            model_wrapper=self.model_wrapper,
            batch_size=batch_size,
            beta=beta,
            epochs=epochs,
            current_round=current_round)

        new_weights: Weights = self.model_wrapper.get_weights()
        new_params = weights_to_parameters(new_weights)

        # save personalized layer to file
        if self.per_layer:
            # self.save_personalization_weight()
            self.model_wrapper.save_personalization_weight(self.cid, self.per_layer, meta_sgd=True)

        # return the refined weights and the number of examples used for training
        return FitRes(
            parameters=new_params,
            num_examples=num_training_sample,
            metrics={'training_loss': training_loss, 'training_accuracy': training_acc}
        )

    def ensemble_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # get config
        weights = parameters_to_weights(ins.parameters)
        config = ins.config
        current_round = int(config["current_round"])
        batch_size = int(config["batch_size"])
        epochs = int(config['epochs'])

        # test the model
        return self.worker.ensemble_test(
            model_wrapper=self.model_wrapper,
            batch_size=batch_size,
            epochs=epochs,
            current_round=current_round,
            weights=weights,
            per_layer=self.per_layer)

    def single_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # get config
        weights = parameters_to_weights(ins.parameters)
        config = ins.config
        current_round = int(config["current_round"])
        batch_size = int(config["batch_size"])
        epochs = int(config['epochs'])

        # assgin personalized layer to local model if the self.per_layer is specified
        if self.per_layer:
            # self.load_personalization_weight(self.cid, weights)
            self.model_wrapper.load_personalization_weight(self.cid, weights, self.per_layer, meta_sgd=True)
        else:
            self.model_wrapper.set_weights(weights)

        # test the model
        return self.worker.test(
            model_wrapper=self.model_wrapper,
            batch_size=batch_size,
            epochs=epochs,
            current_round=current_round)

    def get_best_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # get config
        weights = parameters_to_weights(ins.parameters)
        config = ins.config
        current_round = int(config["current_round"])
        batch_size = int(config["batch_size"])
        epochs = int(config['epochs'])

        return self.worker.best_test(
            model_wrapper=self.model_wrapper,
            batch_size=batch_size,
            epochs=epochs,
            current_round=current_round,
            weights=weights,
            per_layer=self.per_layer)
