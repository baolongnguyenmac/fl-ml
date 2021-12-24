import flwr as fl
from typing import Dict
import argparse

from strategy_server.fed_avg import MyFedAvg

DEFAULT_SERVER_ADDRESS = "localhost:5000"

def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--server_address", type=str, default=DEFAULT_SERVER_ADDRESS, help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds of federated learning (default: 1)")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs of inner task (default: 1)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of inner task (default: 32)")
    parser.add_argument("--fraction_fit", type=float, default=0.3, help="Fraction of available clients used for fit (default: 0.3)")
    parser.add_argument("--fraction_eval", type=float, default=0.3, help="Fraction of available clients used for evaluate (default: 0.3)")
    parser.add_argument("--min_fit_clients", type=int, default=2, help="Minimum number of clients used for fit (default: 2)")
    parser.add_argument("--min_eval_clients", type=int, default=2, help="Minimum number of clients used for evaluate (default: 2)")
    parser.add_argument("--min_available_clients", type=int, default=2, help="Minimum number of available clients required for sampling (default: 2)")
    parser.add_argument("--alpha", type=float, default=0.01, help="Meta-learning rate for FedMeta algorithms (default: 0.01)")
    parser.add_argument("--beta", type=float, default=0.001, help="Meta-learning rate for FedMeta algorithms (default: 0.001)")

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure("server", host=None)

    strategy = MyFedAvg(
        args.fraction_fit,
        args.fraction_eval,
        args.min_fit_clients,
        args.min_eval_clients,
        args.min_available_clients,
        on_fit_config_fn=generate_config(args),
        on_evaluate_config_fn=generate_config(args)
    )

    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy
    )

    # strategy.visualize_result(args)

def generate_config(args):  
    """Returns a function of parameters based on arguments"""
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "current_round": str(rnd),
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
            "beta": str(args.beta),
            "alpha": str(args.alpha) # alpha is used for both fedmeta sgd and fedavg as learning rate of a client
        }
        return config

    return fit_config 

if __name__ == "__main__":
    main()