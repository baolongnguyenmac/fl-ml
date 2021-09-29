import flwr as fl 
from flwr.server.strategy import Strategy

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class FedMetaSGD(Strategy):
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return super().initialize_parameters(client_manager)