import torch

def fedavg(global_params, local_model, client_size, train_size):
    with torch.no_grad():
        for key, value in local_model.named_parameters():
            global_params[key] += (client_size / train_size) * value

    return global_params