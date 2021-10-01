import sys 
sys.path.insert(0, '../')
from client.fedavg_client import FedAvgClient
from client.fedmetamaml_client import FedMetaMAMLClient
from client.fedmetasgd_client import FedMetaSGDClient

from model import model as models, femnist_model, shakespeare_model, sent140_model, meta_sgd_model

import argparse
import torch 
import flwr as fl

DEFAULT_SERVER_ADDRESS = "localhost:5000"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main() -> None:
    """Load data, create and start Client."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )

    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )

    parser.add_argument(
        "--alpha", type=float, required=True, help="Learning rate for fast adaption progress"
    )

    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default).",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=models.FED_AVG,
        help=f"Strategy for client ({models.FED_AVG}, {models.FED_AVG_META}, {models.FED_META_MAML}, {models.FED_META_SDG}).",
    )

    parser.add_argument(
        "--model",
        type=str,
        help=f"{models.FEMNIST_MODEL}, {models.SHAKESPEARE_MODEL}, {models.SENT140_MODEL}"
    )

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Start client
    print(f'Starting client {args.cid}')
    if args.strategy == models.FED_META_SDG:
        model = models.Model(meta_sgd_model.MetaSGD(get_model(args), lr=args.alpha), args.model)
    else:
        model = models.Model(get_model(args), args.model)
    client = get_client(args, model)
    fl.client.start_client(args.server_address, client)

def get_model(args):
    if args.model == models.FEMNIST_MODEL:
        return femnist_model.Femnist()
    elif args.model == models.SHAKESPEARE_MODEL:
        return shakespeare_model.Shakespeare()
    elif args.model == models.SENT140_MODEL:
        return sent140_model.Sent140()
    else:
        print("wrong model syntax")
        return None

def get_client(args, model: models.Model):
    if args.strategy == models.FED_AVG:
        return FedAvgClient(args.cid, model)
    elif args.strategy == models.FED_AVG_META:
        pass
    elif args.strategy == models.FED_META_MAML:
        return FedMetaMAMLClient(args.cid, model)
    elif args.strategy == models.FED_META_SDG:
        return FedMetaSGDClient(args.cid, model)


if __name__ == "__main__":
    main()
