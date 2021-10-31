import torch
import torch.nn as nn

class Sent140(nn.Module):
    def __init__(self, hidden_size: int = 256, embedding_dim: int = 300) -> None:
        super(Sent140, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,  
        )
        self.decoder = nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=x.float()
        _, (h_n, _) = self.lstm(x)  
        pred = self.decoder(h_n[-1])
        return torch.sigmoid(pred)