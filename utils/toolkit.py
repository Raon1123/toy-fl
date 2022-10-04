import copy

import torch
import torch.nn as nn

def random_pmf(num_clients):
    pmf = [1.0 / num_clients] * num_clients
    return pmf


def get_last_param(model):
    """
    Get last parameter of model
    """
    for name, param in model.named_parameters():
        if name[-7:] == '.weight':
            last_weight = copy.deepcopy(param).detach().cpu()
        elif name[-5:] == '.bias':
            last_bias = copy.deepcopy(param).unsqueeze(1).detach().cpu()

    last_param = torch.cat([last_weight, last_bias], dim=1)
    last_param = torch.flatten(last_param)
    return last_param


def get_pairdistance(vec1, vec2):
    """
    """
    pdist = nn.PairwiseDistance(p=2)
    ret = pdist(vec1, vec2)

    return ret
