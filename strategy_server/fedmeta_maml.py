from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
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

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from flwr.server.strategy.fedavg import FedAvg

class FedMetaMAML(FedAvg):
    """Configurable FedMetaMAML strategy implementation."""

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        accept_failures: bool = True,
        alpha: float = 0.01,
        beta: float = 0.01,
        initial_parameters: Optional[Parameters] = None,
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
            initial_parameters=initial_parameters,
        )
        self.alpha = alpha
        self.beta = beta
        self.pre_weights: Optional[Weights] = None
    
    def __repr__(self) -> str:
        # pylint: disable=line-too-long
        rep = f"FedMetaMAML(learning_rate={self.beta}, "
        rep += f"alpha={self.alpha}, pre_weights={self.pre_weights})"

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        weights = parameters_to_weights(parameters)
        self.pre_weights = weights
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]    

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using gradient to update weights."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.pre_weights is None:
            raise Exception("FedMetaMAML pre_weights are None in aggregate_fit")
        
        weights_before = self.pre_weights

        # Convert results
        weights_results = [
            parameters_to_weights(fit_res.parameters)
            for _, fit_res in results
        ]
        total_client = len(results)
        
        weights_prime: Weights = [
            self.beta * reduce(np.add, layer_updates) / total_client
            for layer_updates in zip(*weights_results)
        ]   
        
        new_weights = [
            x - y
            for x, y in zip(weights_before, weights_prime)
        ]

        return weights_to_parameters(new_weights), {}