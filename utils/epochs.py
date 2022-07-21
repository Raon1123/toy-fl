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


def test_epoch(model, dataloader, device='cuda:0', use_pbar=False):
    if use_pbar:
        pbar = tqdm(dataloader, desc='Test epoch')
    else:
        pbar = dataloader

    total, correct = 0, 0
    loss = 0.0

    with torch.no_grad():
        for (imgs, labels) in pbar:
            imgs, labels = imgs.to(device), labels.to(device) 

            outputs = model(imgs)
            _, pred= torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total

    return acc, loss


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

        total_loss += loss.item()

    return total_loss


def run_round(model, 
    partition, 
    datasets, 
    labels,
    args):

    device = args.device
    lossf = nn.CrossEntropyLoss()

    params = {} # parameter for model
    with torch.no_grad():
        for key, value in model.named_parameters():
            params[key] = copy.deepcopy(value)
            params[key].zero_()

    loss_list = []
    total_loss = 0.0
    
    if args.active_algorithm != 'Random':
        pbar = tqdm(partition, desc='Local learning')
        for client in pbar:
            part = np.array(client)
            client_data = datasets[part, :]
            client_label = labels[part]

            train_dataset = TensorDataset(client_data, client_label)
            data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

            acc, loss = test_epoch(model, data_loader, args, device)

            loss_list.append(loss)
            total_loss += loss

        pdf = list(map(lambda item: item/total_loss, loss_list))
    else:
        pdf = [1.0 / args.num_clients] * args.num_clients
    selected_clients = np.random.choice(len(partition), args.active_selection, p=pdf)

    total_size = 0
    for client in selected_clients:
        part = len(partition[client])
        total_size += part

    for client in selected_clients:
        part = np.array(partition[client])
        client_data = datasets[part, :]
        client_label = labels[part]
        client_size = len(part)

        train_dataset = TensorDataset(client_data, client_label)
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        copy_model = copy.deepcopy(model)
        copy_model.to(device)
        optimizer = optim.SGD(copy_model.parameters(), lr=args.lr, momentum=args.momentum)

        for _ in range(args.local_epoch):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device) 
                optimizer.zero_grad()
                outputs = copy_model(x)
                loss = lossf(outputs, y)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            for key, value in copy_model.named_parameters():
                params[key] += (client_size / total_size) * value
    
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(params[key])