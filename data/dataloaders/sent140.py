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
"""Creates a PyTorch Dataset for Leaf Sent140."""
import pickle
from pathlib import Path
import torch 
from flwr.dataset.utils.common import XY
from torch.utils.data import Dataset, DataLoader

class Sent140Dataset(Dataset[XY]):  
    """Creates a PyTorch Dataset for Leaf Shakespeare.
    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset
    """

    def __init__(self, path_to_pickle: Path):

        with open(path_to_pickle, "rb") as open_file:
            data = pickle.load(open_file)
            self.sentence = data["x"]
            self.label = data["y"]
            self.index = data["idx"]
            self.char = data["character"]


    def __len__(self) -> int:
        
        return len(self.label)

    def __getitem__(self, idx: int) -> XY:
        sentence_indices = torch.tensor(self.sentence[idx])
        label = torch.tensor(float(self.label[idx]))
        label = torch.unsqueeze(label, -1)
        return sentence_indices, label

def get_loader(path_to_pickle, batch_size=32, shuffle=True):
    dataset = Sent140Dataset(path_to_pickle)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader, len(dataset)