import torch.nn as nn
import flwr as fl
import argparse
from learn2learn.algorithms.maml import MAML

from client.fedmeta_maml_client import FedMetaMAMLClient
from client.fedmeta_sgd_client import FedMetaSGDClient
from client.fedavg_client import FedAvgClient
from model.cifar_model import Cifar
from model.femnist_model import Femnist
from model.mnist_model import Mnist
from model.model_wrapper import MetaSGDModelWrapper, ModelWrapper, FED_AVG, FED_META_MAML, FED_AVG_META, FED_META_SGD, FEMNIST_MODEL, MNIST_MODEL

DEFAULT_SERVER_ADDRESS = "localhost:5000"

def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--server_address", type=str, default=DEFAULT_SERVER_ADDRESS, help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})")
    parser.add_argument("--cid", type=str, required=True, help="Client ID")
    parser.add_argument("--num_eval_clients", type=int, required=True, help="Num clients for eval")
    parser.add_argument("--alpha", type=float, default=0.01, help="Meta-learning rate for FedMeta algorithms (default: 0.01)")
    parser.add_argument("--strategy_client", type=str, required=True, help="FedAvg, FedMetaMAML, FedAvgMeta, FedMetaSGD")
    parser.add_argument("--model", type=str, required=True, help="sent140, shakespeare, femnist")
    parser.add_argument("--mode", type=str, required=True, help="train or test")
    parser.add_argument("--per_layer", type=int, required=False, help="number of personalized layers (count from the buttom)")

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=None)

    # create and start client
    client = get_client(args, args.cid, get_model(args))
    fl.client.start_client(args.server_address, client)

def get_client(args, cid, model: nn.Module) -> fl.client.Client:
    strategy = args.strategy_client
    if strategy == FED_AVG:
        return FedAvgClient(ModelWrapper(model, args.model), cid, args.mode, args.num_eval_clients, False, args.per_layer)
    elif strategy == FED_AVG_META:
        return FedAvgClient(ModelWrapper(model, args.model), cid, args.mode, args.num_eval_clients, True, args.per_layer)
    elif strategy == FED_META_MAML:
        # return FedMetaMAMLClient(ModelWrapper(model, args.model), cid, args.mode, args.num_eval_clients)
        return FedMetaMAMLClient(ModelWrapper(MAML(model, args.alpha), args.model), cid, args.mode, args.num_eval_clients, args.per_layer)
    elif strategy == FED_META_SGD:
        return FedMetaSGDClient(ModelWrapper(MetaSGDModelWrapper(model, args.alpha), args.model), cid, args.mode, args.num_eval_clients, args.per_layer)

def get_model(args) -> nn.Module:
    if args.model == FEMNIST_MODEL:
        return Femnist()
    elif args.model == MNIST_MODEL:
        return Mnist()
    elif args.model == 'cifar':
        return Cifar()

if __name__ == '__main__':
    main()