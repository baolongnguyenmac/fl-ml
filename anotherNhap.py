import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

def sample_points(k):
    x = torch.rand(k, 1)
    y = torch.randint(low=0, high=2, size=(k,), dtype=int)

    train_x = x[:int(0.8*k)]
    train_y = y[:int(0.8*k)]
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=10)

    test_x = x[int(0.8*k):]
    test_y = y[int(0.8*k):]
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=10)

    return train_loader, test_loader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear(x)
        return self.softmax(out)


NUM_OF_EPOCHS = 10  # epoch of meta training phase
NUM_OF_TASK = 2  # batch of tasks
LOSS_FN = torch.nn.functional.cross_entropy
OUTER_LR = 0.01  # beta

class MetaSGD(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.beta = 0.0001
        self.model = model
        alpha = [torch.ones_like(p) * 0.1 for p in self.model.parameters()]
        alpha = nn.ParameterList([nn.Parameter(lr) for lr in alpha])
        self.alpha = alpha
        self.count = 0 # use for fast adapt

    def forward(self, x):
        return self.model(x)

    def _adapt_update(self, model: nn.Module, grads):
        # Update the params
        if len(list(model._modules)) == 0 and len(list(model._parameters)) != 0:
            for param_key in model._parameters:
                # p = copy.deepcopy(model._parameters[param_key])
                p = model._parameters[param_key].detach().clone()
                # if p is not None:
                #     model._parameters[param_key] = p - p._lr * p.grad
                model._parameters[param_key] = p - self.alpha[self.count] * grads[self.count]
                self.count += 1

        # Then, recurse for each submodule
        for module_key in model._modules:
            model._modules[module_key] = self._adapt_update(model._modules[module_key], grads)
        return model

    def adapt(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        self._adapt_update(self.model, grads)

def training_step(model, batch):
    x, y = batch[0], batch[1]
    pred = model(x)
    loss = LOSS_FN(pred, y)
    return loss

def meta_train(meta_learner: MetaSGD):
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(meta_learner.parameters(), lr=0.1)
    for e in range(NUM_OF_EPOCHS):
        meta_loss = torch.tensor(0.)
        weight_copy = copy.deepcopy(meta_learner.model.state_dict())
        for t in range(NUM_OF_TASK):
            train_loader, test_loader = sample_points(100)

            for batch in train_loader:
                loss = training_step(meta_learner, batch)
                meta_learner.adapt(loss)
                meta_learner.count = 0
                # print('epoch', e, 'adapt', t, ':', list(meta_learner.model.parameters()), '\n')

            for batch in test_loader:
                loss = training_step(meta_learner, batch)
                meta_loss += loss 

            meta_learner.model.load_state_dict(weight_copy)

        meta_loss /= NUM_OF_TASK
        # grad = torch.autograd.grad(meta_loss, meta_learner.parameters())
        # print(grad)
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()
        break

if __name__ == "__main__":
    model = MetaSGD(Model())
    print('model ban dau', list(model.parameters()), '\n')

    train_loader, test_loader = sample_points(100)
    # for x, y in train_loader:
    #     print(x.shape,'\n')
    #     print(y.shape, '\n')
    #     out = model(x)
    #     print(out.shape, '\n')
    #     loss = torch.nn.functional.cross_entropy(out, y)
    #     print(loss, '\n')
    #     break

    meta_train(model)
    print('final', list(model.parameters()))
    # for key in model._modules:
    #     ps = model._modules[key]
    #     lps = list(ps.parameters())
    #     for p in lps:
    #         print(p)
    #         print('grad', p._grad, '\n')


