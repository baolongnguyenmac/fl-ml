import torch
import torch.nn as nn

class MAMLTrain:
    def __init__(self, model: nn.Module, lossFn, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        self.model = model
        self.lossFn = lossFn
        self.optimizer = optimizer
        self.device = device

    def training_step(self, batch):
        """Perform a training step: forward + calculate loss

        Args:
            batch (tuple): a batch of data

        Returns:
            float: loss of batch training
        """
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        preds = self.model(features)
        loss = self.lossFn(preds, labels)
        return loss

    def trainOnSupport(self, supportloader: torch.utils.data.DataLoader, epochs: int) -> None:
        """Train the network using MAML

        Args:
            net (nn.Module): model
            trainloader (torch.utils.data.DataLoader): trained dataloader
            epochs (int): number of epochs
            device (torch.device): train model on device
        """
        print(f"Training {epochs} epoch(s) on {len(supportloader)} batch(es) using {self.device}")

        for e in range(epochs):
            for batch in supportloader:
                # set grad to 0
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()

    def trainOnQuey(self, queryloader: torch.utils.data.DataLoader):
        """Train the network using convention machine learning

        Args:
            net (nn.Module): model
            testloader (torch.utils.data.DataLoader): trained dataloader
            device (torch.device): train model on device
        """
        loss=0
        for batch in queryloader:
            loss += self.training_step(batch)
            
        grad = torch.autograd.grad(loss, self.model.parameters())
        return grad
