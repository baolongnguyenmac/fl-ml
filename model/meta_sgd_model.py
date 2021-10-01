import torch 
import torch.nn as nn 

class MetaSGD(nn.Module):
    def __init__(self, model: nn.Module, lr=0.01):
        super().__init__()
        self.model = model
        alpha = [torch.ones_like(p) * lr for p in self.model.parameters()]
        alpha = nn.ParameterList([nn.Parameter(lr) for lr in alpha])
        self.alpha = alpha
        self.count = 0

    def forward(self, x):
        return self.model(x)

    def _adapt_update(self, model: nn.Module, grads):
        # Update the params
        if len(list(model._modules)) == 0 and len(list(model._parameters)) != 0:
            for param_key in model._parameters:
                p = model._parameters[param_key].detach().clone()
                model._parameters[param_key] = p - self.alpha[self.count] * grads[self.count]
                self.count += 1

        # Then, recurse for each submodule
        for module_key in model._modules:
            model._modules[module_key] = self._adapt_update(model._modules[module_key], grads)
        return model

    def adapt(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        self._adapt_update(self.model, grads)