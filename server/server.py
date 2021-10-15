import argparse
from typing import Dict

import torch
import flwr as fl

import sys 
sys.path.insert(0, '../')

DEFAULT_SERVER_ADDRESS = "localhost:5000"

def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs each client will fine-tune (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used when training each client (default: 32)",
    )   
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for FedAvg",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        help="Alpha for MAML, Meta-SGD or learning rate",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Beta for MAML, Meta-SGD",
    )

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.sample_fraction,
            fraction_eval= args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=generate_config(args),
            on_evaluate_config_fn=generate_config(args)
        )

    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )

def generate_config(args):  
    """Returns a funtion of parameters based on arguments"""
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "learning_rate": str(args.learning_rate),
            "alpha": str(args.alpha),
            "beta": str(args.beta),
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
        }
        return config

    return fit_config 

if __name__ == "__main__":
    main()

