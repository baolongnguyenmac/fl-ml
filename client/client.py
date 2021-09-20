import sys 
sys.path.append('../')
from client import fedavg_client

import argparse
import torch 
import flwr as fl

DEFAULT_SERVER_ADDRESS = "[::]:8080"
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

    # parser.add_argument(
    #     "--log_host",
    #     type=str,
    #     help="Logserver address (no default).",
    # )

    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy for client (no default).",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        help="Server learning rate alpha",
    )

    parser.add_argument(
        "--beta",
        type=float,
        help="Client learning rate beta",
    )

    parser.add_argument(
        "--model",
        type=str,
        help=""
    )

    # parser.add_argument(
    #     "--exp_name",
    #     type=str,
    #     help="Useful experiment name for tensorboard plotting.",
    # )

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    model = dataset.load_model()
    model.to(DEVICE)
    trainset, testset = dataset.load_data()
    print(f'Loading data for client {args.cid}')
    trainset, testset = dataset.load_local_partitioned_data(cid=int(args.cid), 
                                                          iid_fraction = args.iid_fraction, 
                                                          num_partitions = args.num_partitions)
    # Start client
    print(f'Starting client {args.cid}')
    client = get_client(args, model, trainset, testset)
    fl.client.start_client(args.server_address, client)
    
def get_client(args, model, trainset, testset):
    if args.strategy=="perFedAvg":
        return PerFedAvgClient(args.cid, model, trainset, testset, f'{args.exp_name}_iid-fraction_{args.iid_fraction}', args.iid_fraction, args.alpha, args.beta)
    elif args.strategy=="perFedAvgHF":
        return PerFedAvgHFClient(args.cid, model, trainset, testset, f'{args.exp_name}_iid-fraction_{args.iid_fraction}', args.iid_fraction, args.alpha, args.beta)
    return DefaultClient(args.cid, model, trainset, testset, f'{args.exp_name}_iid-fraction_{args.iid_fraction}', args.iid_fraction, args.alpha)
    
    


if __name__ == "__main__":
    main()
