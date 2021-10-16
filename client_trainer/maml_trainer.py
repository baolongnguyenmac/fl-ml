import torch
from torch.utils.data.dataloader import DataLoader

import copy

import sys
sys.path.insert(0, '../')
from model.model_wrapper import ModelWrapper


class MAMLTrainer:
    def __init__(self, model: ModelWrapper, lossFn, optimizerSup: torch.optim.Optimizer, optimizerQuery: torch.optim.Optimizer, device: torch.device, cid: int) -> None:
        self.model = model
        self.lossFn = lossFn
        self.optimizerSup = optimizerSup
        self.optimizerQuery = optimizerQuery
        self.device = device
        self.cid = cid

    def _training_step(self, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        preds = self.model.model(features)
        loss = self.lossFn(preds, labels)
        return loss

    def train(self, support_loader: DataLoader, query_loader: DataLoader, epochs: int) -> None:
        print(f"[Client {self.cid}]: Running {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}")

        w_t_copy = copy.deepcopy(self.model.get_weights())
        for _ in range(epochs):
            for batch in support_loader:
                # set grad to 0
                self.optimizerSup.zero_grad()
                loss = self._training_step(batch)
                loss.backward()
                self.optimizerSup.step()

        for batch in query_loader:
            self.optimizerQuery.zero_grad()
            loss = self._training_step(batch)
            loss.backward()
            self.model.set_weights(w_t_copy)
            self.optimizerQuery.step()

        print(f'[Client {self.cid}] loss: LOSS')
