import torch 

import warnings
warnings.filterwarnings("ignore")

from .base_worker import BaseTester, BaseTrainer
from model.model_wrapper import ModelWrapper, MetaSGDModelWrapper

class MetaSGDTrainer(BaseTrainer):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, epochs: int, beta: float) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size, epochs)
        self.beta = beta

    def train(self):
        print(f'[Client {self.cid}]: Fit in round {self.current_round}')

        support_loader, _ = self.get_loader(support=True)
        query_loader, num_query_sample = self.get_loader(support=False)

        print(f'[Client {self.cid}]: Fit {self.epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        learner: MetaSGDModelWrapper = self.model_wrapper.model.module.clone()
        opt = torch.optim.Adam(self.model_wrapper.model.parameters(), lr=self.beta)

        for epoch in range(self.epochs):
            for batch in support_loader:
                loss, _ = self._training_step(learner, batch)
                learner.adapt(loss)

        training_loss = 0.
        training_acc = 0.
        for batch in query_loader:
            loss, acc = self._training_step(learner, batch)
            training_loss += loss
            training_acc += acc

        opt.zero_grad()
        training_loss /= len(query_loader)
        training_acc /= num_query_sample
        training_loss.backward()
        opt.step()

        print(f'[Client {self.cid}]: Training loss = {float(training_loss)}, Training acc = {float(training_acc)}')
        return float(training_loss), float(training_acc), num_query_sample

class MetaSGDTester(BaseTester):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, num_eval_clients: int, mode: str, epochs: int) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size, num_eval_clients, mode)
        self.epochs = epochs

    def test(self):
        print(f'[Client {self.cid}]: Eval in round {self.current_round}')

        support_loader, _ = self.get_loader(support=True)
        query_loader, num_query_sample = self.get_loader(support=False)
        learner: MetaSGDModelWrapper = self.model_wrapper.model.module.clone()

        print(f'[Client {self.cid}]: Evaluate {self.epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        for e in range(self.epochs):
            for batch in support_loader:
                loss, _ = self._training_step(learner, batch)
                learner.adapt(loss)

        val_loss = 0.
        val_acc = 0.
        for batch in query_loader:
            loss, acc = self._valid_step(learner, batch)
            val_loss += loss
            val_acc += acc

        val_loss /= len(query_loader)
        val_acc /= num_query_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(val_acc)}')
        return float(val_loss), float(val_acc), num_query_sample
