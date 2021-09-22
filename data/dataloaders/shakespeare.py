import pickle
from pathlib import Path
from typing import List

import torch 
from flwr.dataset.utils.common import XY
from torch.utils.data import Dataset, DataLoader

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)

LABEL = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
)

class ShakespeareDataset(Dataset[XY]):  
    """Creates a PyTorch Dataset for Leaf Shakespeare.
    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset
    """

    def __init__(self, path_to_pickle: Path):

        self.characters: str = LEAF_CHARACTERS
        self.num_letters: int = len(self.characters)  # 80

        with open(path_to_pickle, "rb") as open_file:
            data = pickle.load(open_file)
            self.sentence = data["x"]
            self.next_word = data["y"]
            self.index = data["idx"]
            self.char = data["character"]

    def filter_label(self, features, labels):
        pass

    def word_to_indices(self, word: str) -> List[int]:
        """Converts a sequence of characters into position indices in the
        reference string `self.characters`.
        Args:
            word (str): Sequence of characters to be converted.
        Returns:
            List[int]: List with positions.
        """
        indices: List[int] = [self.characters.find(c) for c in word]
        return indices

    def one_hot(self, index, size):
        '''returns one-hot vector with given size and value 1 at given index
        '''
        vec = [0 for _ in range(size)]
        vec[int(index)] = 1
        return vec

    def __len__(self) -> int:
        return len(self.next_word)

    def __getitem__(self, idx: int) -> XY:
        sentence_indices = torch.tensor(self.word_to_indices(self.sentence[idx]))
        # next_word_index = torch.tensor(self.one_hot(self.characters.find(self.next_word[idx]), len(LEAF_CHARACTERS)))
        next_word_index = torch.tensor(self.characters.find(self.next_word[idx]))
        return sentence_indices, next_word_index

def get_loader(path_to_pickle, batch_size=32, shuffle=True):
    dataset = ShakespeareDataset(path_to_pickle)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader, len(dataset)

# loader, _ = get_loader('../shakespeare/test/0/query.pickle')
# for x, y in loader:
#     print(x.shape)
#     print(y.shape)
#     print(y)
#     break