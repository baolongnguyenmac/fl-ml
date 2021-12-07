import torch 
import torch.nn as nn 
import random
random.seed(42)

from model.model_wrapper import ModelWrapper, FEMNIST_MODEL, MNIST_MODEL, CIFAR_MODEL
from data.dataloaders.femnist import get_loader as f_loader
from data.dataloaders.mnist import get_loader as mn_loader
from data.dataloaders.cifar import get_loader as ci_loader

class BaseWorker:
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int) -> None:
        self.model_wrapper = model_wrapper
        self.device = device
        self.cid = cid
        self.current_round = current_round
        self.loader = self._get_loader()
        self.batch_size = batch_size
        self.loss_fn = nn.functional.cross_entropy

    def _get_loader(self):
        if self.model_wrapper.model_name == FEMNIST_MODEL:
            return f_loader
        elif self.model_wrapper.model_name == MNIST_MODEL:
            return mn_loader
        elif self.model_wrapper.model_name == CIFAR_MODEL:
            return ci_loader

    def _training_step(self, model: nn.Module, batch):
        features, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = model(features)
        loss = self.loss_fn(outputs, labels)       
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).sum()

        return loss, acc

    def _valid_step(self, model: nn.Module, batch):
        with torch.no_grad():
            return self._training_step(model, batch)

class BaseTrainer(BaseWorker):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, epochs: int) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size)
        self.epochs = epochs

    def get_loader(self, support: bool):
        s = 'support' if support else 'query'
        id_set = self.cid

        return self.loader(path_to_pickle=f'./data/{self.model_wrapper.model_name}/train/{id_set}/{s}.pickle', batch_size=self.batch_size, shuffle=True)

class BaseTester(BaseWorker):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, num_eval_clients: int, mode: str) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size)
        self.num_eval_clients = num_eval_clients
        self.mode = mode # mode = {'val', 'test'}

    def get_loader(self, support: bool):
        s = 'support' if support else 'query'
        # id_set = random.choice(list(range(self.num_eval_clients)))
        id_set = self.cid
        
        return self.loader(path_to_pickle=f'./data/{self.model_wrapper.model_name}/{self.mode}/{id_set}/{s}.pickle', batch_size=self.batch_size, shuffle=True)
