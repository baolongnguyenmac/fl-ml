import torch 
import torch.nn as nn
from learn2learn.algorithms.meta_sgd import MetaSGD, clone_module, clone_parameters, meta_sgd_update
from learn2learn.algorithms.maml import MAML
import random
random.seed(42)
import plotly.graph_objects as go
import numpy as np

import sys 
sys.path.insert(0, '../')
from data.dataloaders.femnist import get_loader as f_loader
from data.dataloaders.sent140 import get_loader as se_loader
from data.dataloaders.shakespeare import get_loader as sh_loader

from model.femnist_model import Femnist
from model.sent140_model import Sent140
from model.shakespeare_model import Shakespeare

'''
hyper parameters: round: 200, epoch: 1, batch client: 24, batch size: 32
                    alpha: 0.01, beta: 0.001, train client: 299, valid client: 37
these parameters are quite good for femnist
'''

# MODEL = 'femnist'
# ROUNDS = 100
# EPOCHS = 1
# TASKS = 6
# BATCH_SIZE = 32
# ALPHA = 0.01
# BETA = 0.001
# LOSS_FN = torch.nn.functional.cross_entropy if MODEL!='sent140' else torch.nn.functional.binary_cross_entropy
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TRAIN_CLIENT = 299
# VALID_CLIENT = 37

MODEL = 'femnist'
ROUNDS = 200
EPOCHS = 1
TASKS = 24
BATCH_SIZE = 32
ALPHA = 0.01
BETA = 0.001
LOSS_FN = torch.nn.functional.cross_entropy if MODEL!='sent140' else torch.nn.functional.binary_cross_entropy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CLIENT = 299
VALID_CLIENT = 37

class MyMetaSGD(MetaSGD):
    def clone(self):
        return MyMetaSGD(clone_module(self.module),
                        lrs=clone_parameters(self.lrs),
                        first_order=self.first_order)

    def adapt(self, loss, first_order=None):
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = torch.autograd.grad(loss,
                                        self.module.parameters(),
                                        retain_graph=second_order,
                                        create_graph=second_order, allow_unused=True)
        self.module = meta_sgd_update(self.module, self.lrs, gradients)

def main():
    if MODEL == 'sent140':
        loader = se_loader
        model = Sent140()
    elif MODEL == 'femnist':
        loader = f_loader
        model = Femnist()
    elif MODEL == 'shakespeare':
        loader = sh_loader
        model = Shakespeare()

    meta_model = MAML(model, lr=ALPHA).to(DEVICE)
    # meta_model = MyMetaSGD(model, lr=ALPHA).to(DEVICE)
    opt = torch.optim.Adam(meta_model.parameters(), lr=BETA)

    train_history = {}
    train_history['acc'] = []
    train_history['loss'] = []

    valid_history = {}
    valid_history['acc'] = []
    valid_history['loss'] = []

    # outer loop
    for round in range(ROUNDS):
        meta_train_loss = 0.
        meta_train_acc = 0. # percent

        meta_valid_loss = 0.
        meta_valid_acc = 0. # percent

        meta_num_train = 0.
        meta_num_valid = 0.

        # train a batch of task
        for task in range(TASKS):
            task_id = random.choice(list(range(TRAIN_CLIENT)))
            learner = meta_model.clone()
            training_loss, training_acc, num_train = train(learner, loader, task_id)
            meta_num_train += num_train
            meta_train_loss += training_loss * num_train
            meta_train_acc += training_acc * num_train

        # meta opt
        meta_train_acc /= meta_num_train # lấy xấp xỉ độ acc
        opt.zero_grad()
        meta_train_loss /= meta_num_train
        meta_train_loss.backward()
        opt.step()

        train_history['acc'].append(float(meta_train_acc))
        train_history['loss'].append(float(meta_train_loss))
        print(f'[Round {round}]: Training loss: {float(meta_train_loss)}, Training acc: {float(meta_train_acc)}')

        # valid a batch of task
        for task in range(TASKS):
            task_id = random.choice(list(range(VALID_CLIENT)))
            learner = meta_model.clone()
            valid_loss, valid_acc, num_valid = valid(learner, loader, task_id)
            meta_num_valid += num_valid
            meta_valid_loss += valid_loss * num_valid
            meta_valid_acc += valid_acc * num_valid

        meta_valid_loss /= meta_num_valid
        meta_valid_acc /= meta_num_valid

        valid_history['acc'].append(float(meta_valid_acc))
        valid_history['loss'].append(float(meta_valid_loss))
        print(f'[Round {round}]: Valid loss: {float(meta_valid_loss)}, Valid acc: {float(meta_valid_acc)}')

    visualize(train_history, valid_history)

def visualize(training_history, valid_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(list(range(ROUNDS))), y=np.array(training_history['loss']),
                            mode='lines+markers',
                            name='training loss'))
    fig.add_trace(go.Scatter(x=np.array(list(range(ROUNDS))), y=np.array(valid_history['loss']),
                            mode='lines+markers',
                            name='valid loss'))
    fig.update_layout(title='Loss',
                        xaxis_title='Round communication',
                        yaxis_title='Loss')

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=np.array(list(range(ROUNDS))), y=np.array(training_history['acc']),
                            mode='lines+markers',
                            name='training acc'))
    fig1.add_trace(go.Scatter(x=np.array(list(range(ROUNDS))), y=np.array(valid_history['acc']),
                            mode='lines+markers',
                            name='valid acc'))
    fig1.update_layout(title='Accuracy',
                        xaxis_title='Round communication',
                        yaxis_title='Accuracy')

    fig.show()
    fig1.show()

# valid a task
def valid(learner: MyMetaSGD, loader, task_id):
    valid_support_loader, _ = loader(path_to_pickle=f'../data/{MODEL}/val/{task_id}/support.pickle', batch_size=BATCH_SIZE, shuffle=True)
    valid_query_loader, num_valid_query = loader(path_to_pickle=f'../data/{MODEL}/val/{task_id}/query.pickle', batch_size=BATCH_SIZE, shuffle=False)

    for e in range(EPOCHS):
        for batch in valid_support_loader:
            loss, _ = training_step(learner, batch)
            learner.adapt(loss)

    valid_loss = 0.
    valid_acc = 0.
    for batch in valid_query_loader:
        loss, acc = valid_step(learner, batch)
        valid_loss += loss
        valid_acc += acc

    return valid_loss/len(valid_query_loader), valid_acc/num_valid_query, num_valid_query

def valid_step(model: nn.Module, batch):
    with torch.no_grad():
        return training_step(model, batch)

# train a task
def train(learner: MyMetaSGD, loader, task_id):
    train_support_loader, _ = loader(path_to_pickle=f'../data/{MODEL}/train/{task_id}/support.pickle', batch_size=BATCH_SIZE, shuffle=True)
    train_query_loader, num_train_query = loader(path_to_pickle=f'../data/{MODEL}/train/{task_id}/query.pickle', batch_size=BATCH_SIZE, shuffle=False)

    for e in range(EPOCHS):
        for batch in train_support_loader:
            loss, _ = training_step(learner, batch)
            learner.adapt(loss)

    training_loss = 0.
    training_acc = 0.
    for batch in train_query_loader:
        loss, acc = training_step(learner, batch)
        training_loss += loss
        training_acc += acc

    return training_loss/len(train_query_loader), training_acc/num_train_query, num_train_query

def training_step(model: nn.Module, batch):
    features, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
    outputs = model(features)
    loss = LOSS_FN(outputs, labels)
    if MODEL == 'sent140':
        preds = torch.round(outputs)
    else:
        _, preds = torch.max(outputs, dim=1)
    acc = (preds == labels).sum()

    return loss, acc

if __name__ == "__main__":
    main()
