import torch
import torch.nn as nn
from flwr.common import Weights
from collections import OrderedDict
from learn2learn.algorithms.meta_sgd import MetaSGD, meta_sgd_update, clone_module, clone_parameters
import pickle

FED_AVG = "FedAvg"
FED_META_MAML = "FedMetaMAML"
FED_AVG_META = "FedAvgMeta"
FED_META_SGD = "FedMetaSGD"

FEMNIST_MODEL = "femnist"
MNIST_MODEL = "mnist"
CIFAR_MODEL = "cifar"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelWrapper:
    """generate a wrapper that wraps a nn.Module
    """
    def __init__(self, model: nn.Module, model_name: str):
        model = nn.DataParallel(model)
        self.model = model.to(DEVICE)
        self.model_name = model_name

    def get_weights(self) -> Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def save_personalization_weight(self, cid:str, per_layer:int, meta_sgd=False):
        weights = self.get_weights()

        if meta_sgd:
            num_weight = len(weights) # ex: 3layer -> num_weight = 12 (3 weight, 3 bias, 6 alpha)

            # save weight
            personalized_weight = weights[-num_weight//2 + per_layer:-num_weight//2]
            with open(f'./personalized_weight/{cid}.pickle', 'wb') as file_weight:
                pickle.dump(personalized_weight, file_weight)
            file_weight.close()

            # save alpha (is contained by weights)
            personalized_alpha = weights[per_layer:]
            with open(f'./personalized_weight/{cid}_alpha.pickle', 'wb') as file_alpha:
                pickle.dump(personalized_alpha, file_alpha)
            file_alpha.close()
        else:
            personalized_weight = weights[per_layer:]
            with open(f'./personalized_weight/{cid}.pickle', 'wb') as file_input:
                pickle.dump(personalized_weight, file_input)
            file_input.close()

    def load_personalization_weight(self, cid:str, weights:Weights, per_layer:int, meta_sgd=False):
        try: # in case the file do not exist
            if meta_sgd:
                # load local weight from file
                num_weight = len(weights) # ex: 3layer -> num_weight = 12 (3 weight, 3 bias, 6 alpha)
                with open(f'./personalized_weight/{cid}.pickle', 'rb') as file_weight:
                    personalized_weight = pickle.load(file_weight)
                    file_weight.close()
                weights[-num_weight//2 + per_layer:-num_weight//2] = personalized_weight
                file_weight.close()

                # load alpha
                with open(f'./personalized_weight/{cid}_alpha.pickle', 'rb') as file_alpha:
                    personalized_alpha = pickle.load(file_alpha)
                    file_alpha.close()
                weights[per_layer:] = personalized_alpha
                file_alpha.close()
            else:
                # load local weight from file
                with open(f'./personalized_weight/{cid}.pickle', 'rb') as file_input:
                    personalized_weight = pickle.load(file_input)
                    file_input.close()
                # assign local weight to the current weight
                weights[per_layer:] = personalized_weight
        except:
            pass

        # set new weight to the model
        self.set_weights(weights)

class MetaSGDModelWrapper(MetaSGD):
    """generate a meta model that wraps a nn.Module
    """
    def clone(self):
        return MetaSGDModelWrapper(clone_module(self.module),
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
