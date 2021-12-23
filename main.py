import torch.nn as nn
import flwr as fl
import argparse
from typing import Dict
from learn2learn.algorithms.maml import MAML

from client.fedmeta_maml_client import FedMetaMAMLClient
from client.fedmeta_sgd_client import FedMetaSGDClient
from client.fedavg_client import FedAvgClient
from model.femnist_model import Femnist
from model.cifar_model import Cifar
from model.mnist_model import Mnist
from model.model_wrapper import MetaSGDModelWrapper, ModelWrapper, FED_AVG, FED_META_MAML, FED_AVG_META, FED_META_SGD, FEMNIST_MODEL, MNIST_MODEL, CIFAR_MODEL
from strategy_server.fed_avg import MyFedAvg

import os

def main():
    # delete all personalized weight
    filenames = os.listdir('./personalized_weight')
    for filename in filenames:
        os.remove(f'./personalized_weight/{filename}')

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--num_clients", type=int, required=True, help="Num clients for training")
    parser.add_argument("--num_eval_clients", type=int, required=True, help="Num clients for eval")
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
    parser.add_argument("--strategy_client", type=str, required=True, help="FedAvg, FedMetaMAML, FedAvgMeta, FedMetaSGD")
    parser.add_argument("--model", type=str, required=True, help="cifar, mnist, femnist")
    parser.add_argument("--mode", type=str, required=True, help="val or test")
    parser.add_argument("--per_layer", type=int, required=False, help="number of personalized layers (count from the buttom)")

    args = parser.parse_args()

    strategy = MyFedAvg(
        fraction_fit=args.fraction_fit,
        fraction_eval= args.fraction_eval,
        min_fit_clients=args.min_fit_clients,
        min_eval_clients=args.min_eval_clients,
        min_available_clients=args.min_available_clients,
        on_fit_config_fn=generate_config(args),
        on_evaluate_config_fn=generate_config(args)
    )

    fl.simulation.start_simulation(
        client_fn=client_fn_config(args),
        num_clients=args.num_clients,
        client_resources={"num_cpus": 4},
        # client_resources={"num_cpus": 2, "num_gpus": 1},
        num_rounds=args.rounds,
        strategy=strategy
    )

    strategy.visualize_result(args)

def generate_config(args):  
    """Returns a function of parameters based on arguments"""
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "current_round": str(rnd),
            "epochs": str(args.epochs), # epochs of fine-tune process at client
            "batch_size": str(args.batch_size), # batch size of support_loader and query_loader
            "beta": str(args.beta),  # beta is used for fedmeta maml and fedmeta sgd as a meta learning rate of client
            "alpha": str(args.alpha) # alpha is used for fedmeta sgd and fedavg as a learning rate of client
        }
        return config

    return fit_config 

def client_fn_config(args):
    # create a single client instance
    def client_fn(cid: str):
        return get_client(args, cid, get_model(args))

    return client_fn

def get_client(args, cid, model: nn.Module) -> fl.client.Client:
    strategy = args.strategy_client
    if strategy == FED_AVG:
        client = FedAvgClient(ModelWrapper(model, args.model), cid, args.mode, args.num_eval_clients, False, args.per_layer)
    elif strategy == FED_AVG_META:
        client = FedAvgClient(ModelWrapper(model, args.model), cid, args.mode, args.num_eval_clients, True, args.per_layer)
    elif strategy == FED_META_MAML:
        client = FedMetaMAMLClient(ModelWrapper(MAML(model, args.alpha, allow_nograd=True), args.model), cid, args.mode, args.num_eval_clients, args.per_layer)
    elif strategy == FED_META_SGD:
        client = FedMetaSGDClient(ModelWrapper(MetaSGDModelWrapper(model, args.alpha), args.model), cid, args.mode, args.num_eval_clients, args.per_layer)

    return client

def get_model(args) -> nn.Module:
    model: nn.Module = None
    if args.model == CIFAR_MODEL:
        model = Cifar()
    elif args.model == FEMNIST_MODEL:
        model = Femnist()
    elif args.model == MNIST_MODEL:
        model = Mnist()

    return model

if __name__ == "__main__":
    main()
