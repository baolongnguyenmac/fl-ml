import flwr as fl
from typing import Dict

import argparse

import sys
sys.path.insert(0, '../')

from strategy_server.fed_avg import MyFedAvg

DEFAULT_SERVER_ADDRESS = "localhost:5000"

def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size of inner task (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs of inner task (default: 1)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Meta-learning rate for FedMetaMAML algorithm (default: 0.01)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.001,
        help="Meta-learning rate for FedMeta algorithms (default: 0.001)",
    )
    parser.add_argument(
        "--fit_sample_fraction",
        type=float,
        default=0.3,
        help="Fraction of available clients used for fit (default: 0.3)",
    )
    parser.add_argument(
        "--eval_sample_fraction",
        type=float,
        default=0.3,
        help="Fraction of available clients used for evaluate (default: 0.3)",
    )

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure("server", host=None)

    strategy = MyFedAvg(
        fraction_fit=args.fit_sample_fraction,
        # fraction_eval= 1,
        fraction_eval= args.eval_sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_eval_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        on_fit_config_fn=generate_config(args),
        on_evaluate_config_fn=generate_config(args)
    )

    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy
    )

def generate_config(args):  
    """Returns a function of parameters based on arguments"""
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
            "beta": str(args.beta),
            "alpha": str(args.beta) # alpha is used for both fedmeta sgd and fedavg as learning rate of a client
        }
        return config

    return fit_config 

if __name__ == "__main__":
    main()
