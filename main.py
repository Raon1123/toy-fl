import os
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.cnn import CNN
from utils.parser import cifar10_dict, argparser
from utils.datasets import CustomDatasets
from utils.epochs import test_epoch, train_epoch

def run_round(model, 
    partition, 
    datasets, 
    labels,
    args,
    acs=10, 
    batch_sz=32):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    lossf = nn.CrossEntropyLoss()
    device = 'cuda:0'

    params = {} # parameter for model
    with torch.no_grad():
        for key, value in model.named_parameters():
            params[key] = copy.deepcopy(value)
            params[key].zero_()

    loss_list = []
    # training *I like random
    """
    pbar = tqdm(partition, desc='Local learning')
    for client in pbar:
        part = np.array(client)
        client_data = datasets[part, :]
        client_label = labels[part]

        train_dataset = CustomDatasets(client_data, client_label, transforms=transform)
        data_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=2)

        copy_model = copy.deepcopy(model)
        copy_model.to(device)
        loss = train_epoch(copy_model, data_loader, args, device)
        loss_list.append(loss)
    """
    # nice sampling
    selected_clients = np.random.choice(len(partition), acs)

    total_size = 0
    for client in selected_clients:
        part = len(partition[client])
        total_size += part

    for client in selected_clients:
        part = np.array(partition[client])
        client_data = datasets[part, :]
        client_label = labels[part]
        client_size = len(part)

        train_dataset = CustomDatasets(client_data, client_label, transforms=transform)
        data_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=0)

        copy_model = copy.deepcopy(model)
        copy_model.to(device)
        optimizer = optim.SGD(copy_model.parameters(), lr=args.lr, momentum=args.momentum)

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


def main():
    args = argparser()

    device = 'cuda:0'
    alpha = 0.2
    num_labels = 10
    num_clients = 100
    num_rounds = 2000
    torch.manual_seed(32)

    writer = SummaryWriter()

    # datasets
    data_PATH = os.path.join(args.data_dir, 'cifar-10-batches-py')
    train_data, train_labels, test_data, test_labels = cifar10_dict(data_PATH)
    train_data = np.reshape(train_data, (-1,32,32,3))
    test_data = np.reshape(test_data, (-1,32,32,3))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CustomDatasets(train_data, train_labels, transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = CustomDatasets(test_data, test_labels, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    train_sz = len(train_labels)

    # make partition
    partition = []
    client_size = np.random.dirichlet([alpha] * num_clients, size=1) # life is RANDOM
    client_dist = np.random.dirichlet([alpha] * 10, size=num_clients) # distribution of client
    
    label_idx = []
    for label in range(num_labels):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx += [idx]

    # sampling
    for client in range(num_clients):
        sample_idx = []
        size = int(train_sz * client_size[0, client]) + 1
        if size > 0:
            sample_dist = client_dist[client]

            for label in range(num_labels):
                sample = int(sample_dist[label] * size)
                sample = np.random.choice(label_idx[label], size).tolist()
                sample_idx += sample

            partition.append(sample_idx)

    # hyperparam
    model = CNN().to(device)

    # train phase

    pbar = tqdm(range(num_rounds), desc='FL round')
    for round in pbar:
        run_round(model, partition, train_data, train_labels, args)

        # test phase
        acc = test_epoch(model, test_loader, device)
        writer.add_scalar('Acc', acc, round)
        print(acc)

    print("=== Centralized ===")
    for epoch in tqdm(range(200)):
        train_epoch(model, train_loader, args)
    

if __name__=='__main__':
    main()