import torch 
import torch.nn as nn 
import random
random.seed(42)

from model.model_wrapper import MNIST_MODEL, CIFAR_MODEL
from data.dataloaders.femnist import get_loader as f_loader
from data.dataloaders.mnist import get_loader as mn_loader
from data.dataloaders.cifar import get_loader as ci_loader

class BaseWorker:
    def __init__(
        self,
        device: torch.device,
        cid: str,
        new_client: bool
    ) -> None:
        self.device = device
        self.cid = cid
        self.loss_fn = nn.functional.cross_entropy
        self.new_client = new_client

    def get_loader(self, support:bool, train:bool, model_name:str, batch_size:int):
        loader = ci_loader if model_name==CIFAR_MODEL else mn_loader if model_name==MNIST_MODEL else f_loader
        t = 'train' if train else 'test_new' if self.new_client else 'test_local'
        s = 'support' if support else 'query'

        return loader(path_to_pickle=f'./data/{model_name}/{t}/{self.cid}/{s}.pickle', batch_size=batch_size, shuffle=True)

    def _training_step(self, model: nn.Module, batch, return_prob=False):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = model(features)
        loss = self.loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).sum()

        if return_prob:
            return loss, outputs, labels
        return loss, acc, labels, preds

    def _valid_step(self, model: nn.Module, batch, return_prob=False):
        with torch.no_grad():
            return self._training_step(model, batch, return_prob)

# from model.mnist_model import Mnist

# worker = BaseWorker(torch.device('cpu'), 1, False)
# loader, num_sample = worker.get_loader(True, False, 'mnist', 32)

# for batch in loader:
#     worker._training_step(Mnist(), batch)
#     break