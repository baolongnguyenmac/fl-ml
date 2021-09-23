import sys 
sys.path.insert(0, '../')
from client.fedavg_client import FedAvgClient
from model import model as models

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
    model = models.Model(args.model, args.strategy)
    client = get_client(args, model)
    fl.client.start_client(args.server_address, client)

def get_client(args, model):
    if args.strategy == models.FED_AVG:
        return FedAvgClient(args.cid, model)
    elif args.strategy == models.FED_AVG_META:
        pass
    elif args.strategy == models.FED_META_MAML:
        pass
    elif args.strategy == models.FED_META_SDG:
        pass 


if __name__ == "__main__":
    main()
