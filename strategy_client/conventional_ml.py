import torch

class ConventionalTrain:
    def __init__(self, model, lossFn, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
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

    def train(self, trainloader: torch.utils.data.DataLoader, epochs: int) -> None:
        """Train the network using convention machine learning

        Args:
            net (nn.Module): model
            trainloader (torch.utils.data.DataLoader): trained dataloader
            epochs (int): number of epochs
            device (torch.device): train model on device
        """
        print(f"Training {epochs} epoch(s) on {len(trainloader)} batch(es) using {self.device}")

        for e in range(epochs):
            training_loss = 0.
            for batch in trainloader:
                # set grad to 0
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()

                # calculate training loss
                training_loss += loss.item()

            if (e+1)%5 == 0:
                print(f"Epoch {e+1}: loss={training_loss}")

class ConventionalTest:
    def __init__(self, model, lossFn, device: torch.device) -> None:
        self.model = model
        self.lossFn = lossFn
        self.device = device

    def valid_step(self, batch):
        with torch.no_grad():
            features, labels = batch[0].to(self.device), batch[1].to(self.device)
            probs = self.model(features)
            loss = self.lossFn(probs, labels)
            _, preds = torch.max(probs, dim=1)
            acc = (preds==labels).sum()
            return loss, acc

    def test(self, testloader: torch.utils.data.DataLoader) -> None:
        """Test the network

        Args:
            testloader (torch.utils.data.DataLoader): test dataloader
        """
        loss = 0.
        acc = 0.
        num_of_sample = 0
        for batch in testloader:
            tmp_loss, tmp_acc = self.valid_step(batch)
            loss += tmp_loss
            acc += tmp_acc
            num_of_sample += len(batch)
        return loss, acc/num_of_sample
