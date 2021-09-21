# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Minimal example on how to start a simple Flower server."""


import argparse
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict

from logging import INFO
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server.grpc_server.grpc_server import start_insecure_grpc_server

import torch
import torchvision

import flwr as fl
  
DEFAULT_SERVER_ADDRESS = "localhost:1000"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        help="Number of epochs each client will train for (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used when training each client (default: 32)",
    )   
    
    parser.add_argument(
        "--strategy",
        type=str,
        default='FED_AVG',
        help="Strategy for server {FED_AVG, FED_META_MAML, FED_META_SDG}",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate for FedAvg",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        help="alpha for MAML, Meta-SGD or learning rate",
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        help="beta for MAML, Meta-SGD",
    )
    
   
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Create strategy
    strategy = get_strategy(args)

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


def get_strategy(args) -> fl.server.strategy.Strategy:
    if args.strategy == "FED_AVG":
        return fl.server.strategy.FedAvg(
            fraction_fit=args.sample_fraction,
            fraction_eval= 1,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=generate_config(args),
            on_evaluate_config_fn=generate_config(args)
        )
    if args.strategy == "FED_META_MAML":
        pass
    if args.strategy == "FED_META_SGD":
        pass

if __name__ == "__main__":
    main()

