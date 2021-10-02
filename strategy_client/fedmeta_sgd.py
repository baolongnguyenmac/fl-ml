import torch 
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.insert(0, '../')
from client.fedmetasgd_client import MetaSGD 

class MetaSGDTrain:
    def __init__(self, model: MetaSGD, loss_fn, device: torch.device) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def training_step(self, batch):
        feature, label = batch[0].to(self.device), batch[1].to(self.device)
        pred = self.model(feature)
        loss = self.loss_fn(pred, label)
        return loss 

    def train(self, support_loader: DataLoader, query_loader: DataLoader, epochs):
        for _ in range(epochs):
            for batch in support_loader:
                loss = self.training_step(batch)
                self.model.adapt(loss)
                self.model.zero_grad()
                self.model.count = 0

        inner_loss = torch.tensor(0.)
        for batch in query_loader:
            loss = self.training_step(batch)
            inner_loss += loss

        print(f"Loss: {inner_loss.item()}")
        return torch.autograd.grad(inner_loss, self.model.parameters())