import torch 
import torch.nn as nn
import copy 

class MetaSGDTrain:
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

    def train(self, support_set: torch.utils.data.DataLoader, query_set: torch.utils.data.DataLoader, epochs: int):
        torch.autograd.set_detect_anomaly(True)
        for _ in range(epochs):
            meta_loss = torch.tensor(0.)
            # weight_copy = copy.deepcopy()