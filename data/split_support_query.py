import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import random
random.seed(42)

def check_between_zero_and_one(value: str):
    """Tests if value is between 0 an 1"""
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(
            f"""Invalid partition fraction {fvalue}. This must be between [0,1]."""
        )
    return fvalue


def save_partition(save_root: Path, user_idx: int, dataset: str, data: Dict[str, str]):
    """Saves partition for specific client
    Args:
        save_root (Path): Root folder where to save partition
        user_idx (int): User ID
        dataset (str): Dataset {train, validation, test}
        data (Dict[str,str]): Dataset {train, validation, test}
    """
    save_dir = save_root / str(user_idx)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{dataset}.pickle", "wb") as save_file:
        pickle.dump(data, save_file)


def process_user(
    json_file: Dict[str, Any],
    user_idx: str,
    user_str: str,
    list_datasets: List[Tuple[str, float]],
    save_root: Path,
):
    """Creates and saves partition for user
    Args:
        json_file (Dict[str, Any]): JSON file containing user data
        user_idx (str): User ID (counter) in string format
        user_str (str): Original User ID
        list_datasets (List[Tuple[str, float]]): List of datasets and relative fractions
        save_root (Path): Root folder where to save the partition
    """
    sentence = json_file["user_data"][user_str]["x"]
    next_char = json_file["user_data"][user_str]["y"]
    temp = list(zip(sentence, next_char))
    random.shuffle(temp)
    sentence, next_char = zip(*temp)
    start_idx = 0

    for split_id, (dataset, fraction) in enumerate(list_datasets):
        end_idx = start_idx + int(fraction * len(sentence))
        if split_id == len(list_datasets) - 1:  # Make sure we use last indices
            end_idx = len(sentence)
        data = {}
        data["idx"] = user_idx
        data["character"] = user_str
        data["x"] = sentence[start_idx:end_idx]
        data["y"] = next_char[start_idx:end_idx]
        start_idx = end_idx

        save_partition(save_root, user_idx, dataset, data)


def split_json_and_save(
    list_datasets: List[Tuple[str, float]],
    path_to_json: Path,
    save_root: Path,
    prev_users_list: Optional[List[str]] = None,
):
    """Splits LEAF generated datasets and creates individual client partitions.
    Args:
        list_datasets (List[Tuple[str, float]]): list containting dataset tags
            and fraction of dataset split.
        path_to_json (Path): Path to LEAF JSON file containing dataset.
        save_root (Path): Root directory where to save the individual client
            partition files.
    """
    users_list: List[str] = []
    new_users: List[str] = []
    listFile = os.listdir(path_to_json)
    user_idx = 0
    for filename in listFile:
        with open(f'{path_to_json.__str__()}/{filename}') as open_file:
            json_file = json.load(open_file)
            if not prev_users_list:
                users_list = json_file["users"]
            else:
                print("Using previous list of users.")
                users_list = prev_users_list

            for user_str in users_list:
                new_users.append(user_str)
                process_user(json_file, user_idx, user_str, list_datasets, save_root)
                user_idx += 1

    return new_users


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Splits a LEAF Shakespeare train dataset into
        train/validation for each client and saves the clients'
        train/val/test dataset in their respective folder."""
    )
    parser.add_argument(
        "--save_train",
        type=str,
        required=True,
        help="""Root folder where partitions will be save as
                {save_train}/client_id/{support,query}.pickle""",
    )
    parser.add_argument(
        "--leaf_train_json",
        type=str,
        required=True,
        help="""Complete path to JSON file containing the generated
                trainset for LEAF Shakespeare.""",
    )
    parser.add_argument(
        "--save_test",
        type=str,
        required=True,
        help="""Root folder where partitions will be save as
                {save_test}/client_id/{support,query}.pickle""",
    )
    parser.add_argument(
        "--leaf_test_json",
        type=str,
        required=True,
        help="""Complete path to JSON file containing the generated
            *testset* for LEAF Shakespeare.""",
    )
    parser.add_argument(
        "--query_frac",
        type=check_between_zero_and_one,
        required=True,
        default=0.2,
        help="Fraction of original trainset that will be used for validation.",
    )


    args = parser.parse_args()

    # Split train dataset into train and validation
    # then save files for each client
    original_train_dataset = Path(args.leaf_train_json)
    support_frac = 1.0 - args.query_frac
    sup_query_datasets = [("support", support_frac), ("query", args.query_frac)]
    existing_users = split_json_and_save(
        list_datasets=sup_query_datasets,
        path_to_json=original_train_dataset,
        save_root=Path(args.save_train),
    )

    original_test_dataset = Path(args.leaf_test_json)
    existing_users = split_json_and_save(
        list_datasets=sup_query_datasets,
        path_to_json=original_test_dataset,
        save_root=Path(args.save_test),
    )


# python split_support_query.py --save_train mnist/train \
#                               --leaf_train_json mnist/train_json \
#                               --save_test mnist/test \
#                               --leaf_test_json mnist/test_json \
#                               --query_frac 0.8

# python split_val_test.py --path_in='mnist/test' --path_out='mnist/val'