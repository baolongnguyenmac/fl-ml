import torch
import torch.nn as nn
import copy
class MAMLTrain:
    def __init__(self, model, lossFn, optimizerSup: torch.optim.Optimizer, optimizerQuery: torch.optim.Optimizer, device: torch.device) -> None:
        self.model = model
        self.lossFn = lossFn
        self.optimizerSup = optimizerSup
        self.optimizerQuery = optimizerQuery
        self.device = device

    def training_step(self, batch):
        """Perform a training step: forward + calculate loss

        Args:
            batch (tuple): a batch of data

        Returns:
            float: loss of batch training
        """
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        preds = self.model.model(features)
        loss = self.lossFn(preds, labels)
        return loss

    def train(self, supportloader: torch.utils.data.DataLoader, queryloader: torch.utils.data.DataLoader, epochs: int) -> None:
        """Train the network using MAML

        Args:
            net (nn.Module): model
            trainloader (torch.utils.data.DataLoader): trained dataloader
            epochs (int): number of epochs
            device (torch.device): train model on device
        """
        print(f"Training {epochs} epoch(s) on {len(supportloader)} batch(es) using {self.device}")

        w_t_copy = copy.deepcopy(self.model.get_weights())
        
        for _ in range(epochs):
            for batch in supportloader:
                # set grad to 0
                self.optimizerSup.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                self.optimizerSup.step()

        for batch in queryloader:
            self.optimizerQuery.zero_grad()
            loss = self.training_step(batch)
            loss.backward()
            self.model.set_weights(w_t_copy)
            self.optimizerQuery.step()
