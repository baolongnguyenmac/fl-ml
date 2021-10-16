import torch.nn as nn

import flwr as fl
import argparse

import sys
sys.path.insert(0, '../')
from client.fedmeta_maml_client import FedMetaMAMLClient
from client.fedmeta_sgd_client import FedMetaSGDClient
from client.fedavg_client import FedAvgClient
from model.femnist_model import Femnist
from model.sent140_model import Sent140
from model.shakespeare_model import Shakespeare
from model.model_wrapper import MetaSGDModelWrapper, ModelWrapper, FED_AVG, FED_META_MAML, FED_AVG_META, FED_META_SDG, FEMNIST_MODEL, SHAKESPEARE_MODEL, SENT140_MODEL


DEFAULT_SERVER_ADDRESS = "localhost:5000"


def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})"
    )
    parser.add_argument(
        "--cid",
        type=int,
        required=True,
        help="Client ID"
    )

    # only use for fedmeta meta sgd algorithm
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Learning rate (for fast adaption progress)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="FedAvg, FedMetaMAML, FedAvgMeta, FedMetaSGD"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="sent140, shakespeare, femnist"
    )

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=None)

    # create and start client
    client = get_client(args, get_model(args))
    fl.client.start_client(args.server_address, client)


def get_client(args, model: nn.Module) -> fl.client.Client:
    client: fl.client.Client = None
    if args.strategy == FED_AVG:
        client = FedAvgClient(ModelWrapper(model, args.model), args.cid)
    elif args.strategy == FED_META_MAML:
        client = FedMetaMAMLClient(ModelWrapper(model, args.model), args.cid)
    elif args.strategy == FED_META_SDG:
        client = FedMetaSGDClient(ModelWrapper(
            MetaSGDModelWrapper(model, lr=args.alpha), args.model), args.cid)

    return client


def get_model(args) -> nn.Module:
    model: nn.Module = None
    if args.model == SENT140_MODEL:
        model = Sent140()
    elif args.model == FEMNIST_MODEL:
        model = Femnist()
    elif args.model == SHAKESPEARE_MODEL:
        model = Shakespeare()

    return model


if __name__ == "__main__":
    main()
