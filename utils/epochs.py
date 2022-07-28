import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

def get_optimizer(model, args):
    optimizer = None

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    assert optimizer is not None
    return optimizer


#@profile
def test_epoch(model, dataloader, device='cuda:0', use_pbar=False):
    """
    Run one test epoch

    Input 
    - model: pytorch model
    - dataloader: dataloader for test
    - device (str, default: cuda:0): running device
    - use_pbar (bool): using progress bar?

    Output
    - acc: accuracy
    - total_loss: total sum of loss
    - avg_loss: total average of loss
    """
    if use_pbar:
        pbar = tqdm(dataloader, desc='Test epoch')
    else:
        pbar = dataloader

    total, correct = 0, 0
    total_loss = 0.0
    lossf = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device) 

            outputs = model(imgs)
            _, pred= torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            loss = lossf(outputs, labels)
            total_loss += loss.detach().cpu().item()

    acc = 100 * correct / total
    avg_loss = total_loss / total

    return acc, total_loss, avg_loss

#@profile
def train_epoch(model, dataloader, args, device='cuda:0', use_pbar=False):
    if use_pbar:
        pbar = tqdm(dataloader, desc='Train epoch')
    else:
        pbar = dataloader

    # total, correct = 0, 0
    total_loss = 0.0

    optimizer = get_optimizer(model, args)
    lossf = nn.CrossEntropyLoss()

    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = lossf(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().item()

    return total_loss


#@profile
def run_round(model, 
    datasets, 
    args):
    """
    Run one FL round

    Input
    - model: FL model
    - datasets: client TensorDatasets

    Output
    - select_clients: selected client
    - train_loss: total training loss
    """

    device = args.device
    lossf = nn.CrossEntropyLoss()
    dataloaders = []

    params = {} # parameter for model
    with torch.no_grad():
        for key, value in model.named_parameters():
            params[key] = copy.deepcopy(value)
            params[key].zero_()

    loss_list = []
    total_loss = 0.0

    for dataset in datasets:
        client_dataloader = DataLoader(dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        dataloaders.append(client_dataloader)
    
    if args.active_algorithm != 'Random':
        #pbar = tqdm(dataloaders, desc='Local loss')
        for data_loader in dataloaders:
            _, _, loss = test_epoch(model, data_loader, device)

            loss = np.exp(loss)
            loss_list.append(loss)
            total_loss += loss
        pdf = list(map(lambda item: item/total_loss, loss_list))
    else:
        pdf = [1.0 / args.num_clients] * args.num_clients

    # Sampling client
    selected_clients = np.random.choice(args.num_clients, args.active_selection, replace=False, p=pdf)
    print("Selected clients: ", selected_clients)

    total_size = 0
    for client_idx in selected_clients:
        client_size = len(datasets[client_idx])
        total_size += client_size

    total_loss = 0.0
    for client_idx in selected_clients:
        train_dataset = datasets[client_idx]
        client_size = len(train_dataset)
        data_loader = dataloaders[client_idx]

        copy_model = copy.deepcopy(model)
        copy_model.to(device)
        optimizer = optim.SGD(copy_model.parameters(), lr=args.lr, momentum=args.momentum)

        for _ in range(args.local_epoch):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device) 
                optimizer.zero_grad()
                outputs = copy_model(x)

                loss = lossf(outputs, y)
                total_loss += loss.detach().cpu().item()
                
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            for key, value in copy_model.named_parameters():
                params[key] += (client_size / total_size) * value
    
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(params[key])

    return selected_clients, total_loss