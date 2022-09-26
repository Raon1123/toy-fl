import torch

def random_pmf(num_clients):
    pmf = [1.0 / num_clients] * num_clients
    return pmf


def get_last_param(model):
    """
    Get last parameter of model
    """
    for name, param in model.named_parameters():
        if name[-7:] == '.weight':
            last_weight = param.detach().cpu()
        elif name[-5:] == '.bias':
            last_bias = param.unsqueeze(1).detach().cpu()

    last_param = torch.cat([last_weight, last_bias], dim=1)
    return last_param