import sys 
sys.path.insert(0, '../')
from client.fedavg_client import FedAvgClient
from model import model as models

import argparse
import torch 
import flwr as fl

DEFAULT_SERVER_ADDRESS = "localhost:1000"
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
        "--num_partitions", 
        type=int, 
        required=True, 
        help="In our case, this is the total number of clients participating during training. The original dataset is partitioned among clients."
    )

    # parser.add_argument(
    #     "--iid_fraction", 
    #     type=float, 
    #     nargs="?", 
    #     const=1.0, 
    #     help="Fraction of data [0,1] that is independent and identically distributed."
    # ) --> default: 80:20

    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default).",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=models.FED_AVG,
        help="Strategy for client (no default).",
    )

    # parser.add_argument(
    #     "--alpha",
    #     type=float,
    #     help="Server learning rate alpha",
    # )

    # parser.add_argument(
    #     "--beta",
    #     type=float,
    #     help="Client learning rate beta",
    # )

    parser.add_argument(
        "--model",
        type=str,
        help=f"{models.FED_AVG}, {models.FED_AVG_META}, {models.FED_META_MAML}, {models.FED_META_SDG}"
    )

    # parser.add_argument(
    #     "--exp_name",
    #     type=str,
    #     help="Useful experiment name for tensorboard plotting.",
    # )

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Start client
    print(f'Starting client {args.cid}')
    model = models.Model(args.model, args.strategy)
    client = get_client(args, model)
    fl.client.start_client(args.server_address, client)

def get_client(args, model, trainset, testset):
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
