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
import json
import re
import numpy as np
import torch 
from flwr.dataset.utils.common import XY
from torch.utils.data import Dataset, DataLoader

VOCAB_DIR = 'embs.json'


class Sent140Dataset(Dataset[XY]):  
    """Creates a PyTorch Dataset for Leaf Shakespeare.
    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset
    """

    def __init__(self, path_to_pickle: Path):
        self.word_emb_arr, self.indd, self.vocab = self.get_word_emb_arr(VOCAB_DIR)

        with open(path_to_pickle, "rb") as open_file:
            data = pickle.load(open_file)
            self.sentence = data["x"]
            self.label = data["y"]
            self.index = data["idx"]
            self.char = data["character"]

    def split_line(self, line):
        '''split given line/phrase into list of words
        Args:
            line: string representing phrase to be split
        
        Return:
            list of strings, with each string representing a word
        '''
        return re.findall(r"[\w']+|[.,!?;]", line)

    def get_word_emb_arr(self, path):
        with open(path, 'r') as inf:
            embs = json.load(inf)
        vocab = embs['vocab']
        word_emb_arr = np.array(embs['emba'])
        indd = {}
        for i in range(len(vocab)):
            indd[vocab[i]] = i
        vocab = {w: i for i, w in enumerate(embs['vocab'])}
        return word_emb_arr, indd, vocab

    def line_to_indices(self, line, word2id, max_words=25):
        '''converts given phrase into list of word indices
        
        if the phrase has more than max_words words, returns a list containing
        indices of the first max_words words
        if the phrase has less than max_words words, repeatedly appends integer 
        representing unknown index to returned list until the list's length is 
        max_words
        Args:
            line: string representing phrase/sequence of words
            word2id: dictionary with string words as keys and int indices as values
            max_words: maximum number of word indices in returned list
        Return:
            indl: list of word indices, one index for each word in phrase
        '''
        unk_id = len(word2id)
        line_list = self.split_line(line) # split phrase in words
        indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
        indl += [unk_id]*(max_words-len(indl))
        emba = []
        for i in range(0, len(indl)):
            emba.append(self.word_emb_arr[indl[i]])
        return emba

    def __len__(self) -> int:
        
        return len(self.label)

    def __getitem__(self, idx: int) -> XY:
        sentence = self.sentence[idx][4]
        sentence_indices = torch.tensor(self.line_to_indices(sentence, self.indd))
        label = torch.tensor(int(self.label[idx]))
        return sentence_indices, label

def get_loader(path_to_pickle, batch_size=32, shuffle=True):
    dataset = Sent140Dataset(path_to_pickle)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader