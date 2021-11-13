import torch
import flwr as fl
from flwr.common.typing import Scalar
from flwr.common import ParametersRes, Weights, weights_to_parameters
from typing import Dict

from model.model_wrapper import ModelWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseClient(fl.client.Client):
    def __init__(self, model_wrapper: ModelWrapper, cid: str, mode: str, num_eval_clients: int) -> None:
        super().__init__()
        self.model_wrapper = model_wrapper
        self.cid = cid
        # self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        # mode bao gồm 'valid' và 'test', sử dụng mode để đổi tập valid/test trong lúc evaluate của client
        self.mode = mode 
        self.num_eval_clients = num_eval_clients

    def get_parameters(self) -> ParametersRes:
        print(f"[Client {self.cid}]: get_parameters")

        weights: Weights = self.model_wrapper.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties
