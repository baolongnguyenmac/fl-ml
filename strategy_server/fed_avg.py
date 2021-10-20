from logging import WARNING
from typing import Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.strategy.aggregate import aggregate

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def weighted_loss_acc_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results]
    weighted_acc = [num_examples * acc for num_examples, _, acc in results]

    return sum(weighted_losses) / num_total_evaluation_examples, sum(weighted_acc) / num_total_evaluation_examples

class MyFedAvg(FedAvg):

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        loss_aggregated, acc_aggregated = weighted_loss_acc_avg(
            [
                (
                    fit_res.num_examples,
                    fit_res.metrics['training_loss'],
                    fit_res.metrics['training_accuracy'],
                )
                for _, fit_res in results
            ]
        )

        print(f'[Round {rnd}]: Training loss: {loss_aggregated}')
        print(f'[Round {rnd}]: Training accuracy: {acc_aggregated}')

        return weights_to_parameters(aggregate(weights_results)), {'training_loss': loss_aggregated, 'training_accuracy': acc_aggregated}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated, acc_aggregated = weighted_loss_acc_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.metrics['acc'],
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {"accuracy": acc_aggregated}
