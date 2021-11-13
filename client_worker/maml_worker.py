import torch
from torch.utils.data.dataloader import DataLoader
import copy

from .base_worker import BaseTester, BaseTrainer
from model.model_wrapper import ModelWrapper

class MAMLTrainer(BaseTrainer):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, epochs: int, alpha: float, beta: float) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size, epochs)
        self.support_optimizer = torch.optim.Adam(self.model_wrapper.model.parameters(), lr=alpha)
        self.query_optimizer = torch.optim.Adam(self.model_wrapper.model.parameters(), lr=beta)

    def train(self):
        print(f'[Client {self.cid}]: Fit in round {self.current_round}')

        support_loader, _ = self.get_loader(support=True)
        query_loader, num_query_sample = self.get_loader(support=False)

        w_t_copy = copy.deepcopy(self.model_wrapper.get_weights())

        print(f'[Client {self.cid}]: Fit {self.epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        for _ in range(self.epochs):
            for batch in support_loader:
                # set grad to 0
                self.support_optimizer.zero_grad()
                loss, _ = self._training_step(self.model_wrapper.model, batch)
                loss.backward()
                self.support_optimizer.step()

        # only set w_t_copy for model in the first batch
        set_weight_copy = True
        training_loss = 0.
        training_acc = 0.
        for batch in query_loader:
            self.query_optimizer.zero_grad()
            loss, acc = self._training_step(self.model_wrapper.model, batch)
            training_loss += loss
            training_acc += acc
            loss.backward()

            # theta = theta - lr * grad(loss(new_theta, query))
            if set_weight_copy:
                self.model_wrapper.set_weights(w_t_copy)
                set_weight_copy = False
            self.query_optimizer.step()

        training_loss /= len(query_loader)
        training_acc /= num_query_sample

        print(f'[Client {self.cid}]: Training loss = {float(training_loss)}, Training acc = {float(training_acc)}')
        return float(training_loss), float(training_acc), num_query_sample

class MAMLTester(BaseTester):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, num_eval_clients: int, mode: str, epochs: int, alpha: float) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size, num_eval_clients, mode)
        self.epochs = epochs
        self.support_optimizer = torch.optim.Adam(self.model_wrapper.model.parameters(), lr=alpha)

    def test(self):
        print(f'[Client {self.cid}]: Eval in round {self.current_round}')

        support_loader, _ = self.get_loader(support=True)
        query_loader, num_query_sample = self.get_loader(support=False)

        for e in range(self.epochs):
            for batch in support_loader:
                self.support_optimizer.zero_grad()
                loss, _ = self._training_step(self.model_wrapper.model, batch)
                loss.backward()
                self.support_optimizer.step()

        valid_loss = 0.
        valid_acc = 0.
        for batch in query_loader:
            loss, acc = self._valid_step(self.model_wrapper.model, batch)
            valid_loss += loss 
            valid_acc += acc

        valid_loss /= len(query_loader)
        valid_acc /= num_query_sample

        print(f'[Client {self.cid}]: Val loss = {float(valid_loss)}, Val acc = {float(valid_acc)}')
        return float(valid_loss), float(valid_acc), num_query_sample
