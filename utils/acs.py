import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils.toolkit as tk
import models.gp as gp

def acs_random(num_clients, selection, pmf=None):
    if pmf is None:
        pmf = np.ones(num_clients) / num_clients

    selected_clients = np.random.choice(num_clients, 
        selection, 
        replace=False, 
        p=pmf)

    return selected_clients


def acs_badge(args, prev_params):
    params = torch.stack(prev_params)
    pdist_mat = torch.full((args.active_selection, args.num_clients), torch.inf)
    selected_clients = []
 
    # using k-means++
    seed_client = np.random.choice(args.num_clients, 
        1, replace=False).item()
    selected_clients.append(seed_client)
    pdist_mat[0:,] = tk.get_pairdistance(params[seed_client,:], params)

    for t in range(1, args.active_selection):
        pmf, _ = pdist_mat.min(dim=0)
        pmf = torch.div(pmf, pmf.sum())

        select = pmf.multinomial(num_samples=1, replacement=False).item()
        selected_clients.append(select)

        pdist_mat[t,:] = tk.get_pairdistance(params[select,:], params)
    
    return selected_clients


def acs_loss(args, prev_losses):
    loss_array = np.array(prev_losses)
    total_loss = np.sum(loss_array)
    pmf = loss_array / total_loss
    
    if args.verbose:
        print("pmf:", pmf)

    selected_clients = np.random.choice(args.num_clients, 
        args.active_selection, 
        replace=False, 
        p=pmf)

    return selected_clients


def acs_afl(args, prev_losses):
    loss_array = np.array(prev_losses)
    loss_array = loss_array - np.max(loss_array)
    loss_array = np.exp(loss_array)
    total_loss = np.sum(loss_array)
    pmf = loss_array / total_loss
    
    if args.verbose:
        print("pmf:", pmf)

    selected_clients = np.random.choice(args.num_clients, 
        args.active_selection, 
        replace=False, 
        p=pmf)

    return selected_clients


def acs_powd(args, size_array, prev_losses):
    """
    """
    # random sample d clients from data size of each client
    total_size = np.sum(size_array)
    norm = lambda x: x / total_size
    pmf = norm(size_array)
    
    if args.verbose:
        print("pmf:", pmf)

    select_d_client = np.random.choice(args.num_clients,
        args.powd,
        replace=False,
        p=pmf)

    # estimate local loss from sampled client 
    prev_d_loss = prev_losses[select_d_client]
    sorted_d_loss = np.argsort(-prev_d_loss)

    # select highest loss clients
    selected_clients = sorted_d_loss[:args.active_selection]
    selected_clients = select_d_client[selected_clients]

    return selected_clients


def acs_topk(args, prev_losses):
    """
    """
    sorted_loss = np.argsort(-prev_losses)
    selected_clients = sorted_loss[:args.active_selection]

    return selected_clients


def gpr_warmup(args, 
    epoch, 
    gpr,
    prev_losses,
    current_loss, 
    gpr_data):

    gpr.update_loss(np.arange(args.num_clients), current_loss)
    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_clients)),1),
                                    np.expand_dims(current_loss-prev_losses,1),
                                    np.ones([args.num_clients,1])],1)
    gpr_data.append(epoch_gpr_data)
    print("Training GPR")
    gp.TrainGPR(gpr,
            gpr_data[max([(epoch-args.gpr_begin-args.group_size+1),0]):epoch-args.gpr_begin+1],
            matrix_lr=1e-2,
            sigma_lr=0.0,
            gamma=args.GPR_gamma,
            max_epoches=args.GPR_Epoch+50,
            verbose=args.verbose)


def gpr_optimal(args, 
    gpr, 
    active_selection, 
    model, 
    datasets, 
    partitions,
    current_loss, 
    gpr_data,
    device='cpu'):

    gpr.update_loss(np.arange(args.num_clients), current_loss)
    gpr.reset_discount()
    #print("Training with Random Selection For GPR Training:")
    
    # select users for update
    loss_list = []
    selected_clients = np.random.choice(range(args.num_clients), active_selection, replace=False)
    
    for client_idx in selected_clients:
        client_dataloader = tk.get_local_dataloader(args, client_idx, partitions, datasets)

        copy_model = copy.deepcopy(model)
        copy_model.to(device)
        optimizer = optim.SGD(copy_model.parameters(), lr=args.lr, momentum=args.momentum)

        lossf = nn.CrossEntropyLoss()
        for _ in range(args.local_epoch):
            for imgs, labels in client_dataloader:
                imgs, labels = imgs.to(device), labels.to(device) 

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = lossf(outputs, labels)

                loss.backward()
                optimizer.step()

    model.eval()
    for client_idx in range(args.num_clients):
        correct = 0
        running_loss = 0.0
        lossf = nn.CrossEntropyLoss()
        client_dataloader = tk.get_local_dataloader(args, client_idx, partitions, datasets)

        with torch.no_grad():
            for imgs, labels in client_dataloader:
                imgs, labels = imgs.to(device), labels.to(device) 

                outputs = model(imgs)
                loss = lossf(outputs, labels)
                _, pred = torch.max(outputs.data, 1)
                correct += (pred == labels).sum().item()

                running_loss += loss.detach().cpu().item() * labels.size(0)
        loss_list.append(running_loss/len(partitions[client_idx]))

    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_clients)),1),
                                    np.expand_dims(np.array(loss_list)-current_loss,1),
                                    np.ones([args.num_clients,1])],1)
    gpr_data.append(epoch_gpr_data)
    print("Training GPR")
    gp.TrainGPR(gpr,
            gpr_data[-math.ceil(args.group_size/args.GPR_interval):],
            matrix_lr=1e-2,
            sigma_lr=0.0,
            gamma=args.GPR_gamma**args.GPR_interval,
            max_epoches=args.GPR_Epoch,
            verbose=args.verbose)

