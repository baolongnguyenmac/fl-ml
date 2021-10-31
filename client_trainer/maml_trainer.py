import torch
from torch.utils.data.dataloader import DataLoader

import copy

import sys
sys.path.insert(0, '../')
from model.model_wrapper import ModelWrapper


class MAMLTrainer:
    def __init__(self, model_wrapper: ModelWrapper, loss_fn, support_optimizer: torch.optim.Optimizer, query_optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        self.model_wrapper = model_wrapper
        self.loss_fn = loss_fn
        self.support_optimizer = support_optimizer
        self.query_optimizer = query_optimizer
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

    def train(self, support_loader: DataLoader, query_loader: DataLoader, epochs: int):
        w_t_copy = copy.deepcopy(self.model_wrapper.get_weights())
        for _ in range(epochs):
            for batch in support_loader:
                # set grad to 0
                self.support_optimizer.zero_grad()
                loss = self._training_step(batch)[0]
                loss.backward()
                self.support_optimizer.step()

        # only set w_t_copy for model in the first batch
        set_weight_copy = True
        training_loss = 0.
        training_acc = 0.
        for batch in query_loader:
            self.query_optimizer.zero_grad()
            loss, acc = self._training_step(batch)
            training_loss += loss
            training_acc += acc
            loss.backward()

            # theta = theta - lr * grad(loss(new_theta, query))
            if set_weight_copy:
                self.model_wrapper.set_weights(w_t_copy)
                set_weight_copy = False
            self.query_optimizer.step()

        return float(training_loss), float(training_acc)
