import copy
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.parser import get_device
from utils.toolkit import get_last_param, get_local_dataloader
from utils.acs import acs_random, acs_loss, acs_badge, acs_powd
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
    - partitions: represent elements of each client
    - args: from arg parser

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

    # calculate size of each client
    size_arr = np.zeros(args.num_clients)
    for client_idx in range(args.num_clients):
        client_size = len(partitions[client_idx])
        size_arr[client_idx] = client_size

    if args.active_algorithm == 'LossSampling' and prev_losses is not None:
        selected_clients = acs_loss(args, prev_losses)
    elif args.active_algorithm == 'GradientBADGE' and prev_params is not None:
        selected_clients = acs_badge(args, prev_params)
    elif args.active_algorithm == 'powd' and prev_losses is not None:
        selected_clients = acs_powd(args, size_arr, prev_losses)
    elif args.active_algorithm == 'FedCor':
        assert NotImplementedError

        # selected_clients = acs_fedcor(args, size_arr, prev_losses)

    else:
        selected_clients = acs_random(args)  

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

    # evaluate each client loss
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

    # calculate the advantage in off-policy

    # test prediction accuracy of GP model

    # train and exploit GPR

    """
    # test prediction accuracy of GP model
    if args.gpr and epoch>args.warmup:
        test_idx = np.random.choice(range(args.num_users), m, replace=False)
        test_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                    np.ones([args.num_users,1])],1)
        pred_idx = np.delete(list(range(args.num_users)),test_idx)
        
        predict_loss,mu_p,sigma_p = gpr.Predict_Loss(test_data,test_idx,pred_idx)
        print("GPR Predict relative Loss:{:.4f}".format(predict_loss))
        predict_losses.append(predict_loss)

    # train and exploit GPR
    if args.gpr:
        if epoch<=args.warmup and epoch>=args.gpr_begin:# warm-up
            gpr_warmup(args, epoch, gpr, gt_global_losses, gpr_data)
        elif epoch>args.warmup and epoch%args.GPR_interval==0:# normal and optimization round
            gpr_optimal(args, epoch, gpr, m, global_model, train_dataset, user_groups, gt_global_losses, gpr_data) 
        else:# normal and not optimization round
            gpr.update_loss(np.concatenate([np.expand_dims(idxs_users,1),
                                        np.expand_dims(epoch_global_losses,1)],1))
            gpr.update_discount(idxs_users,args.discount)
            
        if epoch>=args.warmup:
            gpr_idxs_users = gpr.Select_Clients(m,args.loss_power,args.epsilon_greedy,args.discount_method,weights,args.dynamic_C,args.dynamic_TH)
            print("GPR Chosen Clients:",gpr_idxs_users)
    """

    return selected_clients, loss_array, param_list