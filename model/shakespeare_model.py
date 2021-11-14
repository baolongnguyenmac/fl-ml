import sys 
sys.path.insert(0, '../')
from data.dataloaders.shakespeare import LEAF_CHARACTERS

import torch
import torch.nn as nn

class Shakespeare(nn.Module):
    def __init__(
        self,
        chars: str = LEAF_CHARACTERS,
        seq_len: int = 80,
        hidden_size: int = 256,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.dict_size = len(chars)
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(self.dict_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,  # Notice batch is first dim now
        )
        # self.decoder = nn.Linear(self.hidden_size, self.dict_size)
        self.decoder = nn.Linear(self.hidden_size, 53) # 26 (a-z) + 26(A-Z) + ' ' = 53

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forwards sentence to obtain next character.

        Args:
            sentence (torch.Tensor): Tensor containing indices of characters

        Returns:
            torch.Tensor: Vector encoding position of predicted character
        """
        encoded_seq = self.encoder(sentence)  # (batch, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(encoded_seq)  # (batch, seq_len, hidden_size)
        pred = self.decoder(h_n[-1])
        return pred
