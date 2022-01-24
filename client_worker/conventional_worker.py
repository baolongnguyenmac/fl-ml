import os
import torch
from flwr.common import Weights

from model.model_wrapper import ModelWrapper
from .base_worker import BaseWorker

class ConventionalWorker(BaseWorker):
    def train(self, model_wrapper:ModelWrapper, batch_size:int, lr:float, epochs:int, current_round:str):
        print(f'[Client {self.cid}]: Fit in round {current_round}')

        support_loader, num_support_sample = self.get_loader(support=True, train=True, model_name=model_wrapper.model_name, batch_size=batch_size)
        query_loader, num_query_sample = self.get_loader(support=False, train=True, model_name=model_wrapper.model_name, batch_size=batch_size)
        train_loader = []
        for batch in support_loader:
            train_loader.append(batch)
        for batch in query_loader:
            train_loader.append(batch)

        num_train_sample = num_support_sample + num_query_sample
        opt = torch.optim.Adam(model_wrapper.model.parameters(), lr=lr)

        print(f'[Client {self.cid}]: Fit {epochs} epoch(s) on {len(train_loader)} batch(es) using {self.device}')
        for e in range(epochs):
            training_loss = 0.
            training_acc = 0.
            for batch in train_loader:
                # set grad to 0
                opt.zero_grad()

                # forward + backward + optimize
                loss, acc = self._training_step(model_wrapper.model, batch)
                loss.backward()
                opt.step()

                # calculate training loss
                training_loss += loss
                training_acc += acc

        training_loss /= len(train_loader)
        training_acc /= num_train_sample

        print(f'[Client {self.cid}]: Training loss = {float(training_loss)}, Training acc = {float(training_acc)}')
        return float(training_loss), float(training_acc), num_train_sample

    def test(self, model_wrapper:ModelWrapper, batch_size:int, current_round:str):
        print(f'[Client {self.cid}]: Eval in round {current_round}')

        val_loader, num_val_sample = self.get_loader(support=False, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)

        print(f'[Client {self.cid}]: Evaluate on {len(val_loader)} batch(es) using {self.device}')
        val_loss = 0.
        val_acc = 0.
        for batch in val_loader:
            tmp_loss, tmp_acc = self._valid_step(model_wrapper.model, batch)
            val_loss += tmp_loss
            val_acc += tmp_acc

        val_loss /= len(val_loader)
        val_acc /= num_val_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(val_acc)}')
        return float(val_loss), float(val_acc), num_val_sample

    def meta_test(self, model_wrapper:ModelWrapper, batch_size:int, lr:float, epochs:int, current_round:str):
        print(f'[Client {self.cid}]: Eval in round {current_round}')

        support_loader, _ = self.get_loader(support=True, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        query_loader, num_query_sample = self.get_loader(support=False, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)

        opt = torch.optim.Adam(model_wrapper.model.parameters(), lr=lr)

        print(f'[Client {self.cid}]: Evaluate {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        for e in range(epochs):
            for batch in support_loader:
                opt.zero_grad()
                loss, _ = self._training_step(model_wrapper.model, batch)
                loss.backward()
                opt.step()

        val_loss = 0.
        val_acc = 0.
        for batch in query_loader:
            loss, acc = self._valid_step(model_wrapper.model, batch)
            val_loss += loss
            val_acc += acc

        val_loss /= len(query_loader)
        val_acc /= num_query_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(val_acc)}')
        return float(val_loss), float(val_acc), num_query_sample

    def ensemble_test(self, model_wrapper:ModelWrapper, batch_size:int, current_round:str, weights:Weights, per_layer:int):
        print(f'[Client {self.cid}]: Ensemble eval in round {current_round}')

        val_loader, num_val_sample = self.get_loader(support=False, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        num_model = len(os.listdir('./personalized_weight'))

        val_loss = 0.
        probs = []
        labels = []
        for i in range(num_model):
            loss = 0. # loss of a model in ensemble schema
            prob = [] # prob (output) array of a client
            model_wrapper.load_personalization_weight(i, weights, per_layer)
            for batch in val_loader:
                tmp_loss, tmp_prob, label = self._valid_step(model_wrapper.model, batch, return_prob=True)
                loss += tmp_loss
                prob.append(tmp_prob)

                if i == 0:
                    labels.append(label)

            # prob array
            prob = torch.cat(prob, dim=0)
            probs.append(prob)

            # loss
            loss /= len(val_loader)
            val_loss += loss

        # loss
        val_loss /= num_model

        # acc
        probs = 1/num_model * sum(probs)
        _, preds = torch.max(probs, dim=1)
        labels = torch.cat(labels, dim=0)
        test_acc = (preds == torch.tensor(labels)).sum() / num_val_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(test_acc)}')
        return float(val_loss), float(test_acc), num_val_sample

    def ensemble_meta_test(self, model_wrapper:ModelWrapper, batch_size:int, lr:float, epochs:int, current_round:str, weights:Weights, per_layer:int):
        print(f'[Client {self.cid}]: Ensemble eval in round {current_round}')

        support_loader, _ = self.get_loader(support=True, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        query_loader, num_query_sample = self.get_loader(support=False, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        num_model = len(os.listdir('./personalized_weight'))

        opt = torch.optim.Adam(model_wrapper.model.parameters(), lr=lr)
        print(f'[Client {self.cid}]: Evaluate {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')

        val_loss = 0.
        probs = []
        labels = []
        for i in range(num_model):
            # load weight and fine tune local model
            model_wrapper.load_personalization_weight(i, weights, per_layer)
            for e in range(epochs):
                for batch in support_loader:
                    opt.zero_grad()
                    loss, _ = self._training_step(model_wrapper.model, batch)
                    loss.backward()
                    opt.step()

            loss = 0. # loss of a model in ensemble schema
            prob = [] # prob (output) array of a client
            for batch in query_loader:
                tmp_loss, tmp_prob, label = self._valid_step(model_wrapper.model, batch, return_prob=True)
                loss += tmp_loss
                prob.append(tmp_prob)

                if i == 0:
                    labels.append(label)

            # prob array
            prob = torch.cat(prob, dim=0)
            probs.append(prob)

            # loss
            loss /= len(query_loader)
            val_loss += loss

        # loss
        val_loss /= num_model

        # acc
        probs = 1/num_model * sum(probs)
        _, preds = torch.max(probs, dim=1)
        labels = torch.cat(labels, dim=0)
        val_acc = (preds == torch.tensor(labels)).sum() / num_query_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(val_acc)}')
        return float(val_loss), float(val_acc), num_query_sample