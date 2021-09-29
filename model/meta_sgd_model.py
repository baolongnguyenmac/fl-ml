import torch 
import torch.nn as nn

class MetaSGD(nn.Module):
    def __init__(self, model: nn.Module, alpha: nn.ParameterList):
        """init meta model

        Args:
            model (nn.Module): femnist, shakespeare or sent140 model
            alpha (nn.ParameterList): learning rate, use for performing element wise with model's weights
        """
        super().__init__()
        self.beta = 0.0001
        self.model = model
        # alpha = [torch.ones_like(p) * 0.1 for p in self.model.parameters()]
        # alpha = nn.ParameterList([nn.Parameter(lr) for lr in alpha])
        self.alpha = alpha
        self.count = 0 # use for fast adapt

    def forward(self, x):
        return self.model(x)

    def _adapt_update(self, model: nn.Module, grads):
        """update weights of model using grad and alpha. alpha has the same model weights' shape

        Args:
            grads (Tuple[Tensor, ...]): loss's grad (using support set)
        """
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
        """perform fast adapt process by calculating grad and calling self._adapt_update func

        Args:
            loss (tensor scalar): loss of model that performs on support set
        """
        grads = torch.autograd.grad(loss, self.model.parameters())
        self._adapt_update(self.model, grads)