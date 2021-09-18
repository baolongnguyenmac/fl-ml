import pickle
from pathlib import Path

import torch 
from flwr.dataset.utils.common import XY
from torch.utils.data import Dataset, DataLoader


class FemnistDataset(Dataset[XY]):  
    """Creates a PyTorch Dataset for Leaf FEMNIST.
    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset
    """

    def __init__(self, path_to_pickle: Path):
        with open(path_to_pickle, "rb") as open_file:
            data = pickle.load(open_file)
            self.imgs = data["x"]
            self.labels = data["y"]
            self.index = data["idx"]
            self.char = data["character"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> XY:
        imgs = torch.tensor(self.imgs[idx])
        labels = torch.tensor(self.labels[idx])
        return imgs, labels

def get_loader(path_to_pickle, batch_size=32, shuffle=True):
    dataset = FemnistDataset(path_to_pickle)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader
