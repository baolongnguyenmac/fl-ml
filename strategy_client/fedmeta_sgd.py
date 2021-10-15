import torch 
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.insert(0, '../')
from model.model import MetaSGD

class MetaSGDTrain:
    def __init__(self, model: MetaSGD, loss_fn, device: torch.device, cid: int) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.cid = cid

    def _training_step(self, model: nn.Module, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        preds = model(features)
        loss = self.loss_fn(preds, labels)
        return loss

    def train(self, support_loader: DataLoader, query_loader: DataLoader, epochs, beta: float):
        learner = self.model.clone()
        opt = torch.optim.Adam(self.model.parameters(), lr=beta)
        print(f'[Client {self.cid}]: Running {epochs} epoch(s) for fast adaption')
        for e in range(epochs):
            for batch in support_loader:
                loss = self._training_step(learner, batch)
                learner.adapt(loss)

        print(f'[Client {self.cid}]: Calculate meta_loss')
        meta_loss = 0.
        for batch in query_loader:
            meta_loss += self._training_step(learner, batch)

        print(f'[Client {self.cid}]: Optimize theta and alpha')
        opt.zero_grad()
        meta_loss.backward()
        opt.step()

        # print(f'[Client {self.cid}]: New weights: {list(self.model.parameters())}')
        print(f'[Client {self.cid}] loss: {meta_loss.item()}')