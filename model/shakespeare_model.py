from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import flwr as fl

DATA_ROOT = "../data/shakespeare"

class Shakespeare(nn.Module):
    def __init__(self) -> None:
        super(Shakespeare, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2), # batch: [#numOfElement, 64, 14, 14]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2), # batch: [#numOfElement, 64, 7, 7]
            
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64*7*7, out_features=2048),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


def load_model():
    """Load Shakespeare model

    Returns:
        Shakespeare: an instance of class Shakespeare
    """
    return Shakespeare()


def train(
    net: Shakespeare,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    net: Shakespeare,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy