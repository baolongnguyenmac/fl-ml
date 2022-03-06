import torch 
import os
from flwr.common import Weights
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

from .base_worker import BaseWorker
from model.model_wrapper import ModelWrapper, MetaSGDModelWrapper

class MetaSGDWorker(BaseWorker):
    def train(self, model_wrapper:ModelWrapper, batch_size:int, beta:float, epochs:int, current_round:str):
        print(f'[Client {self.cid}]: Fit in round {current_round}')

        support_loader, _ = self.get_loader(support=True, train=True, model_name=model_wrapper.model_name, batch_size=batch_size)
        query_loader, num_query_sample = self.get_loader(support=False, train=True, model_name=model_wrapper.model_name, batch_size=batch_size)

        print(f'[Client {self.cid}]: Fit {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        learner: MetaSGDModelWrapper = model_wrapper.model.module.clone()
        opt = torch.optim.Adam(model_wrapper.model.parameters(), lr=beta)

        for epoch in range(epochs):
            for batch in support_loader:
                loss, _, _, _ = self._training_step(learner, batch)
                learner.adapt(loss)

        training_loss = 0.
        training_acc = 0.
        for batch in query_loader:
            loss, acc, _, _ = self._training_step(learner, batch)
            training_loss += loss
            training_acc += acc

        opt.zero_grad()
        training_loss /= len(query_loader)
        training_acc /= num_query_sample
        training_loss.backward()
        opt.step()

        print(f'[Client {self.cid}]: Training loss = {float(training_loss)}, Training acc = {float(training_acc)}')
        return float(training_loss), float(training_acc), num_query_sample

    def test(self, model_wrapper:ModelWrapper, batch_size:int, epochs:int, current_round:str):
        print(f'[Client {self.cid}]: Eval in round {current_round}')

        support_loader, _ = self.get_loader(support=True, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        query_loader, num_query_sample = self.get_loader(support=False, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        learner: MetaSGDModelWrapper = model_wrapper.model.module.clone()

        print(f'[Client {self.cid}]: Evaluate {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        for e in range(epochs):
            for batch in support_loader:
                loss, _, _, _ = self._training_step(learner, batch)
                learner.adapt(loss)

        val_loss = 0.
        val_acc = 0.
        labels = []
        preds = []
        for batch in query_loader:
            loss, acc, label, pred = self._valid_step(learner, batch)
            val_loss += loss
            val_acc += acc
            labels.extend(label)
            preds.extend(pred)
        
        if (model_wrapper.model_name == 'cifar' and current_round == 600) or (model_wrapper.model_name == 'mnist' and current_round == 300):
            labels = [int(x) for x in labels]
            preds = [int(x) for x in preds]

            for i in range(len(preds)):
                if preds[i] not in set(labels):
                    preds[i] = (set(labels) - set([labels[i]])).pop()

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

        val_loss /= len(query_loader)
        val_acc /= num_query_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(val_acc)}')
        return float(val_loss), float(val_acc), num_query_sample, float(precision), float(recall), float(f1)

    def ensemble_test(self, model_wrapper:ModelWrapper, batch_size:int, epochs:int, current_round:str, weights:Weights, per_layer:int):
        print(f'[Client {self.cid}]: Ensemble eval in round {current_round}')

        support_loader, _ = self.get_loader(support=True, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        query_loader, num_query_sample = self.get_loader(support=False, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        num_model = len(os.listdir('./personalized_weight'))//2

        print(f'[Client {self.cid}]: Evaluate {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        val_loss = 0.
        probs = []
        labels = []
        for i in range(num_model):
            # load weight and fine tune local model
            model_wrapper.load_personalization_weight(i, weights, per_layer, meta_sgd=True)
            learner:MetaSGDModelWrapper = model_wrapper.model.module.clone()
            for e in range(epochs):
                for batch in support_loader:
                    loss, _, _, _ = self._training_step(learner, batch)
                    learner.adapt(loss)

            loss = 0. # loss of a model in ensemble schema
            prob = [] # prob (output) array of a client
            for batch in query_loader:
                tmp_loss, tmp_prob, label = self._valid_step(learner, batch, return_prob=True)
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

    def best_test(self, model_wrapper:ModelWrapper, batch_size:int, epochs:int, current_round:str, weights:Weights, per_layer:int):
        print(f'[Client {self.cid}]: Get best eval in round {current_round}')

        support_loader, _ = self.get_loader(support=True, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        query_loader, num_query_sample = self.get_loader(support=False, train=False, model_name=model_wrapper.model_name, batch_size=batch_size)
        num_model = len(os.listdir('./personalized_weight'))//2

        print(f'[Client {self.cid}]: Evaluate {epochs} epoch(s) on {len(support_loader)} batch(es) using {self.device}')
        # find the per-layer that fit best to the new data (produce min loss)
        losses = []
        for i in range(num_model):
            # load weight and fine tune local model
            model_wrapper.load_personalization_weight(i, weights, per_layer, meta_sgd=True)
            learner:MetaSGDModelWrapper = model_wrapper.model.module.clone()
            for e in range(epochs):
                finetune_loss = 0.
                for batch in support_loader:
                    loss, _, _, _ = self._training_step(learner, batch)
                    learner.adapt(loss)

                    finetune_loss += loss

            finetune_loss /= len(support_loader)
            losses.append(finetune_loss)

        # get min loss index
        min_idx = min(range(len(losses)), key = lambda i: losses[i])

        # load min-loss weight and fine tune local model
        model_wrapper.load_personalization_weight(min_idx, weights, per_layer, meta_sgd=True)
        learner:MetaSGDModelWrapper = model_wrapper.model.module.clone()
        for e in range(epochs):
            finetune_loss = 0.
            for batch in support_loader:
                loss, _, _, _ = self._training_step(learner, batch)
                learner.adapt(loss)

        # test the model
        val_loss = 0.
        val_acc = 0.
        labels = []
        preds = []
        for batch in query_loader:
            loss, acc, label, pred = self._valid_step(learner, batch)
            val_loss += loss
            val_acc += acc
            labels.extend(label)
            preds.extend(pred)

        if (model_wrapper.model_name == 'cifar' and current_round == 600) or (model_wrapper.model_name == 'mnist' and current_round == 300):
            labels = [int(x) for x in labels]
            preds = [int(x) for x in preds]

            for i in range(len(preds)):
                if preds[i] not in set(labels):
                    preds[i] = (set(labels) - set([labels[i]])).pop()
                    
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

        val_loss /= len(query_loader)
        val_acc /= num_query_sample

        print(f'[Client {self.cid}]: Val loss = {float(val_loss)}, Val acc = {float(val_acc)}')
        return float(val_loss), float(val_acc), num_query_sample, float(precision), float(recall), float(f1)
