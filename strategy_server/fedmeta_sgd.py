from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from functools import reduce

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvg

class FedMetaSGD(FedAvg):
    def __init__(
        self, 
        fraction_fit: float = 0.1, 
        fraction_eval: float = 0.1, 
        min_fit_clients: int = 2, 
        min_eval_clients: int = 2, 
        min_available_clients: int = 2, 
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]] = None, 
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None, 
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None, 
        accept_failures: bool = True, 
        initial_parameters: Optional[Parameters] = None,
        beta: float = 0.001
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit, 
            fraction_eval=fraction_eval, 
            min_fit_clients=min_fit_clients, 
            min_eval_clients=min_eval_clients, 
            min_available_clients=min_available_clients, 
            eval_fn=eval_fn, 
            on_fit_config_fn=on_fit_config_fn, 
            on_evaluate_config_fn=on_evaluate_config_fn, 
            accept_failures=accept_failures, 
            initial_parameters=None
        )
        self.beta = beta

    def configure_fit(
        self, 
        rnd: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        weights = parameters_to_weights(parameters)
        print(f'\n\nweight and theta from a client: {weights[-1][0]}\n\n')
        self.pre_weights = weights
        return super().configure_fit(rnd, parameters, client_manager)

    def aggregate_fit(
        self, 
        rnd: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print("\nCalculate global model's weights")

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        grads_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples) 
            for client, fit_res in results
        ]

        total_client = len(results)

        # sum_of_grads
        total_grads = grads_results[0][0]
        for i in range(1, len(grads_results)):
            grads, _ = grads_results[i]
            for i in range(len(total_grads)):
                try:
                    total_grads[i] += grads[i]
                except Exception as e:
                    print(e)
                    pass

        print(f'\n\ntotal grads: {total_grads[-1][0]}')

        # beta/#client * sum_of_grads
        for i in range(len(total_grads)):
            try:
                total_grads[i] *= self.beta/total_client
            except Exception as e:
                print(e)
                pass

        new_weights = []
        for i in range(len(self.pre_weights)):
            try:
                new_weights.append(self.pre_weights[i] - total_grads[i])
            except Exception as e:
                print(e)
                new_weights.append(self.pre_weights[i])

        print('\n\nnew_weights:', new_weights[-1][0], '\n\n')

        return weights_to_parameters(new_weights), {}

    # def aggregate_fit(
    #     self, 
    #     rnd: int, 
    #     results: List[Tuple[ClientProxy, FitRes]], 
    #     failures: List[BaseException]
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     if not results:
    #         return None, {}
    #     # Do not aggregate if there are failures and failures are not accepted
    #     if not self.accept_failures and failures:
    #         return None, {}
    #     # Convert results
    #     grads_results = [
    #         (parameters_to_weights(fit_res.parameters), fit_res.num_examples) 
    #         for client, fit_res in results
    #     ]

    #     total_client = len(results)

    #     # sum_of_grads
    #     total_grads = grads_results[0][0]
    #     for i in range(1, len(grads_results)):
    #         weights, _ = grads_results[i]
    #         for i in range(len(total_grads)):
    #             total_grads[i] += weights[i]

    #     # beta/#client * sum_of_grads
    #     for i in range(len(total_grads)):
    #         total_grads[i] *= self.beta/total_client

    #     new_weights = [
    #         x - y
    #         for x, y in zip(self.pre_weights, total_grads)
    #     ]

    #     print(f'\n\nNew weights: {new_weights}\n\n')

    #     return weights_to_parameters(new_weights), {}