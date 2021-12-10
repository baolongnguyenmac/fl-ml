import torch

from model.model_wrapper import ModelWrapper
from .base_worker import BaseTrainer, BaseTester

class ConventionalTrainer(BaseTrainer):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, epochs: int, lr: float) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size, epochs)
        self.lr = lr

    def train(self):
        print(f'[Client {self.cid}]: Fit in round {self.current_round}')
        support_loader, num_support_sample = self.get_loader(True)
        query_loader, num_query_sample = self.get_loader(False)
        train_loader = []
        for batch in support_loader:
            train_loader.append(batch)
        for batch in query_loader:
            train_loader.append(batch)
        num_train_sample = num_support_sample + num_query_sample
        opt = torch.optim.Adam(self.model_wrapper.model.parameters(), lr=self.lr, weight_decay=0.01)
        # opt = torch.optim.SGD(self.model_wrapper.model.parameters(), lr=self.lr)

        print(f'[Client {self.cid}]: Fit {self.epochs} epoch(s) on {len(train_loader)} batch(es) using {self.device}')
        for e in range(self.epochs):
            training_loss = 0.
            training_acc = 0.
            for batch in train_loader:
                # set grad to 0
                opt.zero_grad()

                # forward + backward + optimize
                loss, acc = self._training_step(self.model_wrapper.model, batch)
                loss.backward()
                opt.step()

                # calculate training loss
                training_loss += loss
                training_acc += acc

        training_loss /= len(train_loader)
        training_acc /= num_train_sample

        print(f'[Client {self.cid}]: Training loss = {float(training_loss)}, Training acc = {float(training_acc)}')
        return float(training_loss), float(training_acc), num_train_sample

class ConventionalTester(BaseTester):
    def __init__(self, model_wrapper: ModelWrapper, device: torch.device, cid: str, current_round: int, batch_size: int, num_eval_clients: int, mode: str, epochs: int = None, lr: float = None) -> None:
        super().__init__(model_wrapper, device, cid, current_round, batch_size, num_eval_clients, mode)
        self.lr = lr
        self.epochs = epochs

    def test(self):
        print(f'[Client {self.cid}]: Eval in round {self.current_round}')
        test_loader, num_test_sample = self.get_loader(False)

        print(f'[Client {self.cid}]: Evaluate on {len(test_loader)} batch(es) using {self.device}')
        test_loss = 0.
        test_acc = 0.
        for batch in test_loader:
            tmp_loss, tmp_acc = self._valid_step(self.model_wrapper.model, batch)
            test_loss += tmp_loss
            test_acc += tmp_acc

        test_loss /= len(test_loader)
        test_acc /= num_test_sample

        print(f'[Client {self.cid}]: Val loss = {float(test_loss)}, Val acc = {float(test_acc)}')
        return float(test_loss), float(test_acc), num_test_sample

    def meta_test(self):
        print(f'[Client {self.cid}]: Eval in round {self.current_round}')
        support_loader, _ = self.get_loader(True)
        query_loader, num_query_sample = self.get_loader(False)
        opt = torch.optim.Adam(self.model_wrapper.model.parameters(), lr=self.lr)

        print(f'[Client {self.cid}]: Evaluate {self.epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        for e in range(self.epochs):
            for batch in support_loader:
                opt.zero_grad()
                loss, _ = self._training_step(self.model_wrapper.model, batch)
                loss.backward()
                opt.step()

        val_loss = 0.
        val_acc = 0.
        for batch in query_loader:
            loss, acc = self._valid_step(self.model_wrapper.model, batch)
            val_loss += loss
            val_acc += acc

        val_loss /= len(query_loader)
        val_acc /= num_query_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(val_acc)}')
        return float(val_loss), float(val_acc), num_query_sample
