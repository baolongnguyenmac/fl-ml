import sys 
sys.path.insert(0, '../')
from model.meta_sgd_model import MetaSGD

import torch 
import torch.nn as nn

class MetaSGDTrain:
    def __init__(self, model: MetaSGD, lossFn, device: torch.device) -> None:
        self.model = model
        self.lossFn = lossFn
        self.device = device

    def training_step(self, batch):
        """Perform a training step: forward + calculate loss

        Args:
            batch (tuple): a batch of data

        Returns:
            float: loss of batch training
        """
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        preds = self.model(features)
        loss = self.lossFn(preds, labels)
        return loss

    def train(self, support_loader: torch.utils.data.DataLoader, query_loader: torch.utils.data.DataLoader, epochs: int):
        """train the model: fast adapt on support set and calculate grad on the loss of query set

        Args:
            support_loader (torch.utils.data.DataLoader): loader of support set
            query_loader (torch.utils.data.DataLoader): loader of query set
            epochs (int): num of epochs for fast adapt

        Returns:
            Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: calculate grad of loss follow by model's weight and alpha
        """

        torch.autograd.set_detect_anomaly(True)
        print(f"Fast adapt {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}")
        for _ in range(epochs):
            for batch in support_loader:
                loss = self.training_step(batch)
                self.model.adapt(loss)
                self.model.zero_grad()
                self.model.count = 0 # reset count for fast adapt

        inner_loss = torch.tensor(0.)
        for batch in query_loader:
            loss = self.training_step(batch)
            inner_loss += loss

        print(f"Loss: {inner_loss.item()}")
        return torch.autograd.grad(inner_loss, self.model.parameters(), retain_graph=True)


# from torch.utils.data import DataLoader, TensorDataset

# def sample_points(k):
#     x = torch.rand(k, 1)
#     y = torch.randint(low=0, high=2, size=(k,), dtype=int)

#     train_x = x[:int(0.8*k)]
#     train_y = y[:int(0.8*k)]
#     train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=10)

#     test_x = x[int(0.8*k):]
#     test_y = y[int(0.8*k):]
#     test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=10)

#     return train_loader, test_loader

# class Meo(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1,2)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         out = self.linear(x)
#         return self.softmax(out)

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Meo()
# alpha = [torch.ones_like(p) * 0.1 for p in model.parameters()]
# alpha = nn.ParameterList([nn.Parameter(lr) for lr in alpha])
# meta_model = MetaSGD(Meo(), alpha)
# trainer = MetaSGDTrain(meta_model, torch.nn.functional.cross_entropy, DEVICE)

# print('before', list(meta_model.model.parameters()), '\n\n')
# support, query = sample_points(100)
# print(trainer.train(support, query, 20))