import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '../')
from model.model_wrapper import MetaSGDModelWrapper

class MetaSGDTrainer:
    def __init__(self, model: MetaSGDModelWrapper, loss_fn, device: torch.device, cid: int) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.cid = cid

    def _training_step(self, model: nn.Module, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = model(features)
        loss = self.loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).sum()
        return loss, acc

    def train(self, support_loader: DataLoader, query_loader: DataLoader, epochs, beta: float):
        learner = self.model.clone()
        opt = torch.optim.Adam(self.model.parameters(), lr=beta)
        for e in range(epochs):
            for batch in support_loader:
                loss = self._training_step(learner, batch)[0]
                learner.adapt(loss)

        print(f'[Client {self.cid}]: Calculate meta_loss and optimize (theta, alpha)')
        training_loss = 0.
        training_acc = 0.
        set_weight_copy = True
        for batch in query_loader:
            opt.zero_grad()
            loss, acc = self._training_step(self.model, batch)
            training_loss += loss
            training_acc += acc
            loss.backward()
            if set_weight_copy:
                self.model.load_state_dict(learner.state_dict())
                set_weight_copy = False
            opt.step()

        # print(f'[Client {self.cid}]: New weights: {list(self.model.parameters())}')
        return float(training_loss), float(training_acc)
