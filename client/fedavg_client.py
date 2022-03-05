import torch
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, Weights, weights_to_parameters, parameters_to_weights

from client_worker.conventional_worker import ConventionalWorker
from .base_client import BaseClient
from model.model_wrapper import ModelWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedAvgClient(BaseClient):
    def __init__(
        self, 
        model_wrapper: ModelWrapper, 
        cid: str, 
        meta:bool,
        per_layer: int = None, 
        new_client: bool = None
    ) -> None:
        super().__init__(model_wrapper, cid, per_layer, new_client)
        self.meta = meta
        self.worker = ConventionalWorker(device=DEVICE, cid=self.cid, new_client=self.new_client)

    def fit(self, ins: FitIns) -> FitRes:
        # get training config
        weights: Weights = parameters_to_weights(ins.parameters)
        config = ins.config
        current_round = int(config['current_round'])
        epochs = int(config['epochs'])
        batch_size = int(config['batch_size'])
        lr = float(config['alpha'])

        # assgin personalized layer to local model if the self.per_layer is specified
        if self.per_layer:
            # self.load_personalization_weight(self.cid, weights)
            self.model_wrapper.load_personalization_weight(self.cid, weights, self.per_layer)
        else:
            self.model_wrapper.set_weights(weights)

        # train model
        training_loss, training_acc, num_train_sample = self.worker.train(
            model_wrapper=self.model_wrapper, 
            batch_size=batch_size, 
            lr=lr, 
            epochs=epochs, 
            current_round=current_round)

        new_weights: Weights = self.model_wrapper.get_weights()
        new_params = weights_to_parameters(new_weights)

        # save personalized layer to file
        if self.per_layer:
            # self.save_personalization_weight()
            self.model_wrapper.save_personalization_weight(self.cid, self.per_layer)

        # return the refined weights and the number of examples used for training
        return FitRes(
            parameters=new_params,
            num_examples=num_train_sample,
            metrics={'training_loss': training_loss, 'training_accuracy': training_acc}
        )

    def ensemble_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.meta:
            # get training config
            config = ins.config
            weights:Weights = parameters_to_weights(ins.parameters)
            current_round = int(config['current_round'])
            batch_size = int(config['batch_size'])
            epochs = int(config['epochs'])
            lr = float(config['alpha'])

            return self.worker.ensemble_meta_test(
                model_wrapper=self.model_wrapper,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                current_round=current_round,
                weights=weights,
                per_layer=self.per_layer)

        else:
            # get training config
            config = ins.config
            weights:Weights = parameters_to_weights(ins.parameters)
            current_round = int(config['current_round'])
            batch_size = int(config['batch_size'])

            return self.worker.ensemble_test(
                model_wrapper=self.model_wrapper,
                batch_size=batch_size,
                current_round=current_round,
                weights=weights,
                per_layer=self.per_layer)

    def single_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.meta:
            # get training config
            weights: Weights = parameters_to_weights(ins.parameters)
            config = ins.config
            current_round = int(config['current_round'])
            batch_size = int(config['batch_size'])
            epochs = int(config['epochs'])
            lr = float(config['alpha'])

            self.model_wrapper.set_weights(weights)

            # test the meta model
            return self.worker.meta_test(
                model_wrapper=self.model_wrapper, 
                batch_size=batch_size, 
                lr=lr, 
                epochs=epochs, 
                current_round=current_round)

        else:
            # get training config
            weights: Weights = parameters_to_weights(ins.parameters)
            config = ins.config
            current_round = int(config['current_round'])
            batch_size = int(config['batch_size'])
            
            self.model_wrapper.set_weights(weights)

            # test the model
            return self.worker.test(
                model_wrapper=self.model_wrapper,
                batch_size=batch_size,
                current_round=current_round)

    def get_best_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.meta:
            # get training config
            config = ins.config
            weights:Weights = parameters_to_weights(ins.parameters)
            current_round = int(config['current_round'])
            batch_size = int(config['batch_size'])
            epochs = int(config['epochs'])
            lr = float(config['alpha'])

            return self.worker.best_test(
                model_wrapper=self.model_wrapper,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                current_round=current_round,
                weights=weights,
                per_layer=self.per_layer)
        else:
            raise NotImplementedError('get_best_evaluate method is not implemented for FedAvg algorithm')
