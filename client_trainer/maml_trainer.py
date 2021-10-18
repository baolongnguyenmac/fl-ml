import torch
from torch.utils.data.dataloader import DataLoader

import copy

import sys
sys.path.insert(0, '../')
from model.model_wrapper import ModelWrapper


class MAMLTrainer:
    def __init__(self, model_wrapper: ModelWrapper, loss_fn, support_optimizer: torch.optim.Optimizer, query_optimizer: torch.optim.Optimizer, device: torch.device, cid: int) -> None:
        self.model_wrapper = model_wrapper
        self.loss_fn = loss_fn
        self.support_optimizer = support_optimizer
        self.query_optimizer = query_optimizer
        self.device = device
        self.cid = cid

    def _training_step(self, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        preds = self.model_wrapper.model(features)
        loss = self.loss_fn(preds, labels)
        return loss

    def train(self, support_loader: DataLoader, query_loader: DataLoader, epochs: int) -> None:
        print(f"[Client {self.cid}]: Running {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}")

        w_t_copy = copy.deepcopy(self.model_wrapper.get_weights())
        for _ in range(epochs):
            for batch in support_loader:
                # set grad to 0
                self.support_optimizer.zero_grad()
                loss = self._training_step(batch)
                loss.backward()
                self.support_optimizer.step()

        # only set w_t_copy for model in the first batch
        set_weight_copy = True
        for batch in query_loader:
            self.query_optimizer.zero_grad()
            loss = self._training_step(batch)
            loss.backward()
            if set_weight_copy:
                self.model_wrapper.set_weights(w_t_copy)
                set_weight_copy = False
            self.query_optimizer.step()

        print(f'[Client {self.cid}] loss: LOSS')
