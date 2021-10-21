import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '../')
from model.model_wrapper import MetaSGDModelWrapper, ModelWrapper

class MetaSGDTrainer:
    def __init__(self, model_wrapper: ModelWrapper, loss_fn, device: torch.device) -> None:
        self.model_wrapper = model_wrapper
        self.loss_fn = loss_fn
        self.device = device

    def _training_step(self, model: nn.Module, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = model(features)
        loss = self.loss_fn(outputs, labels)
        if self.model_wrapper.model_name == 'sent140':
            preds = torch.round(outputs)
        else:
            _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).sum()

        return loss, acc

    def train(self, support_loader: DataLoader, query_loader: DataLoader, epochs, beta: float):
        learner: MetaSGDModelWrapper = self.model_wrapper.model.clone()
        opt = torch.optim.Adam(self.model_wrapper.model.parameters(), lr=beta)

        for epoch in range(epochs):
            for batch in support_loader:
                loss, _ = self._training_step(learner, batch)
                learner.adapt(loss)

        training_loss = 0.
        training_acc = 0.
        for batch in query_loader:
            loss, acc = self._training_step(learner, batch)
            training_loss += loss
            training_acc += acc

        opt.zero_grad()
        training_loss.backward()
        opt.step()

        return float(training_loss), float(training_acc)