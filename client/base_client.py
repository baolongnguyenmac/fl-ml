import flwr as fl
from flwr.common.typing import Scalar
from flwr.common import ParametersRes, Weights, weights_to_parameters, EvaluateIns, EvaluateRes
from typing import Dict

from model.model_wrapper import ModelWrapper

class BaseClient(fl.client.Client):
    def __init__(self, model_wrapper: ModelWrapper, cid: str, per_layer:int=None, new_client:bool=None) -> None:
        super().__init__()
        self.model_wrapper = model_wrapper
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.per_layer = (-1) * per_layer * 2 if per_layer is not None else None
        self.new_client = new_client

    def get_parameters(self) -> ParametersRes:
        print(f"[Client {self.cid}]: get_parameters")

        weights: Weights = self.model_wrapper.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_best_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        raise NotImplementedError('get_best_evaluate method of base client has not been override yet')

    def ensemble_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        raise NotImplementedError('ensemble_evaluate method of base client has not been override yet')

    def single_evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        raise NotImplementedError('single_evaluate method of base client has not been override yet')

    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #     if self.new_client and self.per_layer:
    #         current_round = int(ins.config['current_round'])
    #         if current_round % 20 == 0 or current_round == 1:
    #             # # run ensemble
    #             # val_loss, val_acc, num_val_sample = self.ensemble_evaluate(ins)

    #             # choose the personalization weight that fit best to the new client
    #             val_loss, val_acc, num_val_sample = self.get_best_evaluate(ins)
    #         else:
    #             return None
    #     else:
    #         val_loss, val_acc, num_val_sample = self.single_evaluate(ins)

    #     return EvaluateRes(
    #         loss=val_loss, num_examples=num_val_sample, metrics={'acc': val_acc}
    #     )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        current_round = int(ins.config['current_round'])
        if current_round % 20 == 0 or current_round == 1:
            if self.new_client and self.per_layer and (self.model_wrapper.strategy=='FedMetaMAML' or self.model_wrapper.strategy=='FedMetaSGD'):
                # # run ensemble
                # val_loss, val_acc, num_val_sample = self.ensemble_evaluate(ins)

                # choose the personalization weight that fit best to the new client
                val_loss, val_acc, num_val_sample, precision, recall, f1  = self.get_best_evaluate(ins)
            else:
                val_loss, val_acc, num_val_sample, precision, recall, f1 = self.single_evaluate(ins)
        
            return EvaluateRes(
                loss=val_loss, num_examples=num_val_sample, metrics={'acc': val_acc, 'precision': precision, 'recall': recall, 'f1': f1}
            )