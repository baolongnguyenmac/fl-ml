import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


class ConventionalTrainer:
    def __init__(self, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer, device: torch.device, cid: int) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.cid = cid

    def _training_step(self, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)
        return loss

    def train(self, trainloader: DataLoader, epochs: int) -> None:
        print(f"[Client {self.cid}]: Running {epochs} epoch(s) on {len(trainloader)} batch(es) using {self.device}")

        training_loss = 0.
        for e in range(epochs):
            for batch in trainloader:
                # set grad to 0
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self._training_step(batch)
                loss.backward()
                self.optimizer.step()

                # calculate training loss
                training_loss += loss.item()

        print(f'[Client {self.cid}] training loss (on support set): {training_loss/epochs}')


class ConventionalTester:
    def __init__(self, model: nn.Module, loss_fn, device: torch.device, cid: int) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.cid = cid

    def _valid_step(self, batch):
        with torch.no_grad():
            features, labels = batch[0].to(
                self.device), batch[1].to(self.device)
            probs = self.model(features)
            loss = self.loss_fn(probs, labels)
            _, preds = torch.max(probs, dim=1)
            acc = (preds == labels).sum()
            return loss, acc

    def test(self, testloader: DataLoader):
        print(f'[Client {self.cid}]: Validate')
        loss = 0.
        acc = 0.
        for batch in testloader:
            tmp_loss, tmp_acc = self._valid_step(batch)
            loss += tmp_loss
            acc += tmp_acc

        print(f'[Client {self.cid}] loss and acc: LOSS, ACC')
        return loss, acc
