# Copyright 2021 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Creates a PyTorch Dataset for Leaf Shakespeare."""
import pickle
from pathlib import Path
from typing import List

import torch 
from flwr.dataset.utils.common import XY
from torch.utils.data import Dataset, DataLoader

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
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
        next_word_index = torch.tensor(self.one_hot(self.characters.find(self.next_word[idx]), len(LEAF_CHARACTERS)))
        return sentence_indices, next_word_index

def get_loader(path_to_pickle, batch_size=1, shuffle=True):
    dataset = ShakespeareDataset(path_to_pickle)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader
