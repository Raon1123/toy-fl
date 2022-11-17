import copy
import random

import numpy as np

import torch
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

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


def get_dataset_labels(dataset):
    try:
        labels = dataset.targets
    except AttributeError:
        labels = dataset.tensors[1]

    return labels
