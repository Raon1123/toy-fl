import copy
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.parser import get_device, get_optimizer
from utils.toolkit import random_pmf, get_last_param


#@profile
def test_epoch(model, dataloader, device, use_pbar=False):
    """
    Run one test epoch

    Input 
    - model: pytorch model
    - dataloader: dataloader for test
    - device
    - use_pbar (bool): using progress bar?

    Output
    - acc: accuracy
    - total_loss: total sum of loss
    """
    if use_pbar:
        pbar = tqdm(dataloader, desc='Test epoch')
    else:
        pbar = dataloader

    model.eval()

    total, correct = 0, 0
    running_loss = 0.0
    lossf = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device) 

            outputs = model(imgs)
            loss = lossf(outputs, labels)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            running_loss += loss.item()

    acc = 100 * correct / total

    return acc, (running_loss / total)


def train_epoch(model, optimizer, lossf, dataloader, args, device, use_pbar=False):
    if use_pbar:
        pbar = tqdm(dataloader, desc='Train epoch')
    else:
        pbar = dataloader

    # total, correct = 0, 0
    running_loss = 0.0

    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = lossf(outputs, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.detach().cpu().item()

    return running_loss

#@profile
def run_round(model, 
    datasets, 
    partitions,
    args,
    prev_losses=None,
    prev_params=None):
    """
    Run one FL round

    Input
    - model: FL model
    - datasets: client TensorDatasets

    Output
    - select_clients: selected client
    - train_loss: total training loss
    """

    lossf = nn.CrossEntropyLoss()
    device = get_device(args)
    dataloaders = []

    params = {} # parameter for model
    with torch.no_grad():
        for key, value in model.named_parameters():
            params[key] = copy.deepcopy(value)
            params[key].zero_()

    loss_list = []
    param_list = []
    total_loss = 0.0
    current_param = get_last_param(model)

    """
    for partition in partitions:
        datasubset = Subset(datasets, partition)
        client_dataloader = DataLoader(datasubset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        dataloaders.append(client_dataloader)
    """
    if args.active_algorithm == 'LossSampling':
        if prev_losses is None:
            pmf = random_pmf(args.num_clients)
        else:
            loss_array = np.exp(prev_losses)
            total_loss = np.sum(loss_array)
            loss_list = loss_array.tolist()  
            pmf = list(map(lambda item: item/total_loss, loss_list))
    elif args.active_algorithm == 'GradientBADGE':
        if prev_params is None:
            pmf = random_pmf(args.num_clients)
        else:
            diff_param_list = [] # shape (N, paramshape)
            for param in prev_params:
                diff_param = current_param - param
                diff_param_list.append(diff_param)
        # Sampling by BADGE
    elif args.active_algorithm == 'Random':
        pmf = random_pmf(args.num_clients)
    else:
        Exception("Wrong active algorithm: "+args.active_algorithm)

    # Sampling client
    selected_clients = np.random.choice(args.num_clients, args.active_selection, replace=False, p=pmf)
    #print("Selected clients: ", selected_clients)

    train_size = 0

    for client_idx in selected_clients:
        client_size = len(partitions[client_idx])
        train_size += client_size

    for client_idx in range(args.num_clients):
        train_partition = partitions[client_idx]
        client_size = len(train_partition)

        datasubset = Subset(datasets, train_partition)
        client_dataloader = DataLoader(datasubset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)

        copy_model = copy.deepcopy(model)
        copy_model.to(device)
        optimizer = optim.SGD(copy_model.parameters(), lr=args.lr, momentum=args.momentum)

        for _ in range(args.local_epoch):
            for data in client_dataloader:
                X, y = data[0].to(device), data[1].to(device) 

                optimizer.zero_grad()
                outputs = copy_model(X)
                loss = lossf(outputs, y)

                loss.backward()
                optimizer.step()

        if args.active_algorithm == 'GradientBADGE':
            last_param = get_last_param(copy_model)
            param_list.append(current_param - last_param)

        # FedAVG
        if client_idx in selected_clients:
            with torch.no_grad():
                for key, value in copy_model.named_parameters():
                    params[key] += (client_size / train_size) * value
    
    # apply value to global model
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(params[key])

    loss_list = []
    for partition in partitions:
        datasubset = Subset(datasets, partition)
        dataloader = DataLoader(datasubset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.num_workers, 
            pin_memory=args.pin_memory)
        _, loss = test_epoch(model, dataloader, device)

        loss = np.exp(loss)
        loss_list.append(loss)

    loss_array = np.array(loss_list)

    return selected_clients, loss_array, param_list