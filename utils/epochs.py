import copy
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils.parser import get_device
from utils.toolkit import get_last_param, get_local_dataloader, get_partition_weight
from utils.acs import (
    acs_random, acs_loss, acs_badge, acs_powd, 
    gpr_warmup, gpr_optimal)
from utils.federated import fedavg


def test_epoch(model, dataloader, device):
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
    model.eval()

    correct = 0
    running_loss = 0.0
    lossf = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device) 

            outputs = model(imgs)
            loss = lossf(outputs, labels)
            _, pred = torch.max(outputs.data, 1)
            correct += (pred == labels).sum().item()

            running_loss += loss.detach().cpu().item() * labels.size(0)

    acc = 100 * correct / len(dataloader.dataset)

    return acc, running_loss / len(dataloader.dataset)


def train_local_epoch(model, optimizer, dataloader, device='cpu'):
    running_loss = 0.0
    lossf = nn.CrossEntropyLoss()

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device) 

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = lossf(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.detach().cpu().item() * labels.size(0)

    return model, running_loss / len(dataloader.dataset)


def run_round(
    communication_round,
    model, 
    datasets, 
    partitions,
    args,
    prev_losses=None,
    prev_params=None,
    gpr=None,
    gpr_data=None):
    """
    Run one FL round

    Input
    - model: FL model
    - datasets: client TensorDatasets
    - partitions: represent elements of each client
    - args: from arg parser
    - prev_losses: losses from previous round
    - prev_params: parameters from previous round
    - gpr: GP model for FedCor

    Output
    - select_clients: selected client
    - train_loss: total training loss
    """
    SEARCH_EVERY_METHOD = ['GradientBADGE']

    device = get_device(args)

    params = {} # parameter for model
    with torch.no_grad():
        for key, value in model.named_parameters():
            params[key] = copy.deepcopy(value)
            params[key].zero_()

    loss_list = []
    param_list = []
    current_param = get_last_param(model)
    weights = get_partition_weight(partitions)

    # calculate size of each client
    size_arr = np.zeros(args.num_clients)
    for client_idx in range(args.num_clients):
        client_size = len(partitions[client_idx])
        size_arr[client_idx] = client_size

    if args.active_algorithm == 'LossSampling' and prev_losses is not None:
        sample_losses = prev_losses

        # cumulate loss sampling
        if args.active_algorithm[-4:] == 'cum':
            sample_losses = prev_losses * size_arr

        selected_clients = acs_loss(args, sample_losses)
    elif args.active_algorithm == 'GradientBADGE' and prev_params is not None:
        selected_clients = acs_badge(args, prev_params)
    elif args.active_algorithm == 'powd' and prev_losses is not None:
        selected_clients = acs_powd(args, size_arr, prev_losses)
    elif args.active_algorithm == 'FedCor':
        if communication_round>=args.warmup:
            selected_clients = gpr.select_clients(
                number=args.active_selection, 
                loss_power=0.3,
                discount_method='time',
                weights=weights 
                )
        else:
            selected_clients = acs_random(args.num_clients, args.active_selection)  
    else:
        pmf = None
        # size sampling
        if args.active_algorithm[-4:] == 'size':
            pmf = size_arr / np.sum(size_arr)
            
        selected_clients = acs_random(args.num_clients, args.active_selection, pmf)  

    train_size = np.sum(size_arr[np.array(selected_clients)])

    if args.active_algorithm in SEARCH_EVERY_METHOD:
        for client_idx in range(args.num_clients):
            client_dataloader = get_local_dataloader(args, client_idx, partitions, datasets)

            copy_model = copy.deepcopy(model)
            copy_model.to(device)
            optimizer = optim.SGD(copy_model.parameters(), lr=args.lr, momentum=args.momentum)

            for _ in range(args.local_epoch):
                copy_model, _ = train_local_epoch(copy_model, optimizer, client_dataloader, device)

            client_last_param = get_last_param(copy_model)
            param_list.append(current_param - client_last_param)

            if client_idx in selected_clients:
                params = fedavg(params, copy_model, 
                    client_size=size_arr[client_idx], 
                    train_size=train_size)
    else:
        for client_idx in selected_clients:
            client_dataloader = get_local_dataloader(args, client_idx, partitions, datasets)

            copy_model = copy.deepcopy(model)
            copy_model.to(device)
            optimizer = optim.SGD(copy_model.parameters(), lr=args.lr, momentum=args.momentum)

            for _ in range(args.local_epoch):
                copy_model, _ = train_local_epoch(copy_model, optimizer, client_dataloader, device)

            params = fedavg(params, copy_model, 
                client_size=size_arr[client_idx], 
                train_size=train_size)
    
    # apply value to global model
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(params[key])

    # evaluate each client loss at current rounds
    loss_list = []
    for partition in partitions:
        datasubset = Subset(datasets, partition)
        dataloader = DataLoader(datasubset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=args.pin_memory)

        _, loss = test_epoch(model, dataloader, device)

        loss_list.append(loss)

    loss_array = np.array(loss_list)
    loss_diff = loss_array - np.array(prev_losses) if prev_losses is not None else loss_array

    # test prediction accuracy of GP model
    if (args.active_algorithm == 'FedCor') and communication_round>args.warmup:
        test_idx = np.random.choice(range(args.num_clients), args.active_selection, replace=False)
        pred_idx = np.delete(list(range(args.num_clients)),test_idx)
        
        predict_loss, _, _ = gpr.predict_loss(
            index=np.arange(args.num_clients),
            value=loss_diff,
            priori_idx=test_idx,
            posteriori_idx=pred_idx)

        print("GPR Predict relative Loss:{:.4f}".format(predict_loss))

    # train and exploit GPR
    if args.active_algorithm == 'FedCor':
        if communication_round<=args.warmup and communication_round>=args.gpr_begin:# warm-up
            gpr_warmup(
                args, 
                communication_round, 
                gpr, 
                prev_losses,
                loss_array,
                gpr_data)
        elif communication_round>args.warmup and communication_round%args.GPR_interval==0:# normal and optimization round
            gpr_optimal(args, 
                gpr, 
                args.active_selection, 
                model, 
                datasets, 
                partitions, 
                loss_array, 
                gpr_data,
                device) 
        else:# normal and not optimization round
            gpr.update_loss(selected_clients, loss_array[selected_clients])
            gpr.update_discount(selected_clients, args.fedcor_beta)
            
    return selected_clients, loss_array, param_list