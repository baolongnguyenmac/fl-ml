from logging import WARNING
from typing import List, Tuple
from flwr.common import FitIns,Parameters,parameters_to_weights
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class MyFedAvg(FedAvg):
    def configure_fit(self, rnd: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        weight = parameters_to_weights(parameters)
        print('\n\n', weight, '\n\n')
        return super().configure_fit(rnd, parameters, client_manager)