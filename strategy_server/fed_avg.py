from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Weights,
    Scalar,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import plotly.graph_objects as go

def weighted_loss_acc_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results]
    weighted_acc = [num_examples * acc for num_examples, _, acc in results]

    return sum(weighted_losses) / num_total_evaluation_examples, sum(weighted_acc) / num_total_evaluation_examples

class MyFedAvg(FedAvg):
    def __init__(self, fraction_fit: float = 0.1, fraction_eval: float = 0.1, min_fit_clients: int = 2, min_eval_clients: int = 2, min_available_clients: int = 2, eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]] = None, on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None, on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None, accept_failures: bool = True, initial_parameters: Optional[Parameters] = None) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_eval=fraction_eval, min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients, min_available_clients=min_available_clients, eval_fn=eval_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters)
        self.training_history = {}
        self.training_history['loss'] = []
        self.training_history['acc'] = []

        self.valid_history = {}
        self.valid_history['loss'] = []
        self.valid_history['acc'] = []

        self.x_axis = []

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

        self.x_axis.append(rnd)

        self.training_history['loss'].append(loss_aggregated)
        self.training_history['acc'].append(acc_aggregated)

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

        self.valid_history['loss'].append(loss_aggregated)
        self.valid_history['acc'].append(acc_aggregated)

        print(f'[Round {rnd}]: Valid loss: {loss_aggregated}')
        print(f'[Round {rnd}]: Valid accuracy: {acc_aggregated}')

        return loss_aggregated, {"accuracy": acc_aggregated}

    def visualize_result(self, args):
        loss_title = f'[LOSS] [{args.model}, {args.strategy_client}]: Clients/round: {args.min_fit_clients} - Epochs: {args.epochs} - Batch size: {args.batch_size} - Alpha: {args.alpha} - Beta = {args.beta}'
        acc_title = f'[ACCURACY] [{args.model}, {args.strategy_client}]: Clients/round: {args.min_fit_clients} - Epochs: {args.epochs} - Batch size: {args.batch_size} - Alpha: {args.alpha} - Beta = {args.beta}'

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.array(self.x_axis), y=np.array(self.training_history['loss']),
                            mode='lines+markers',
                            name='training loss'))
        fig.add_trace(go.Scatter(x=np.array(self.x_axis), y=np.array(self.valid_history['loss']),
                            mode='lines+markers',
                            name='valid loss'))
        fig.update_layout(title=loss_title,
                        xaxis_title='Round communication',
                        yaxis_title='Loss')

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.array(self.x_axis), y=np.array(self.training_history['acc']),
                            mode='lines+markers',
                            name='training acc'))
        fig1.add_trace(go.Scatter(x=np.array(self.x_axis), y=np.array(self.valid_history['acc']),
                            mode='lines+markers',
                            name='valid acc'))
        fig1.update_layout(title=acc_title,
                        xaxis_title='Round communication',
                        yaxis_title='Accuracy')

        fig.show()
        fig1.show()
