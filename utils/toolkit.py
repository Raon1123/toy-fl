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
    return last_param


def get_similarity(args, vec1, vec2):
    assert vec1.shape == vec2.shape

    vec1 = torch.flatten(vec1)
    vec2 = torch.flatten(vec2)
    vec2 = vec2 / torch.norm(vec2)

    if args.similarity_measure == 'distance':
        ret = (vec1 - vec2).pow(2).sum().sqrt().item()
    elif args.similarity_measure == 'similar':
        cos = nn.CosineSimilarity(dim=0)
        ret = cos(vec1, vec2).abs().item()

    return ret
