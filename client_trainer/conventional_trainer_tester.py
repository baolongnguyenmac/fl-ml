import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.insert(0, '../')
from model.model_wrapper import ModelWrapper

class ConventionalTrainer:
    def __init__(self, model_wrapper: ModelWrapper, loss_fn, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        self.model_wrapper = model_wrapper
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def _training_step(self, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.model_wrapper.model(features)
        loss = self.loss_fn(outputs, labels)
        if self.model_wrapper.model_name == 'sent140':
            preds = torch.round(outputs)
        else:
            _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).sum()

        return loss, acc

    def train(self, trainloader: DataLoader, epochs: int) -> None:
        for e in range(epochs):
            training_loss = 0.
            training_acc = 0.
            for batch in trainloader:
                # set grad to 0
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss, acc = self._training_step(batch)
                loss.backward()
                self.optimizer.step()

                # calculate training loss
                training_loss += loss
                training_acc += acc

        return float(training_loss), float(training_acc)


class ConventionalTester:
    def __init__(self, model_wrapper: ModelWrapper, loss_fn, device: torch.device) -> None:
        self.model_wrapper = model_wrapper
        self.loss_fn = loss_fn
        self.device = device

    def _valid_step(self, batch):
        with torch.no_grad():
            features, labels = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model_wrapper.model(features)
            loss = self.loss_fn(outputs, labels)
            if self.model_wrapper.model_name == 'sent140':
                preds = torch.round(outputs)
            else:
                _, preds = torch.max(outputs, dim=1)
            acc = (preds == labels).sum()

            return loss, acc

    def test(self, testloader: DataLoader):
        loss = 0.
        acc = 0.
        for batch in testloader:
            tmp_loss, tmp_acc = self._valid_step(batch)
            loss += tmp_loss
            acc += tmp_acc

        return float(loss), float(acc)
