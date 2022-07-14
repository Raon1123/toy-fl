import os
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.cnn import CNN
from utils.parser import cifar10_dict, argparser, apply_transform
from utils.epochs import test_epoch, train_epoch

def run_round(model, 
    partition, 
    datasets, 
    labels,
    args):

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
    total_loss = 0.0
    # training *I like random
    
    pbar = tqdm(partition, desc='Local learning')
    for client in pbar:
        part = np.array(client)
        client_data = datasets[part, :]
        client_label = labels[part]

        train_dataset = TensorDataset(client_data, client_label)
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        copy_model = copy.deepcopy(model)
        copy_model.to(device)
        loss = train_epoch(copy_model, data_loader, args, device)

        loss_list.append(loss)
        total_loss += loss
    # nice sampling
    selected_clients = np.random.choice(len(partition), args.active_selection)

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


def main():
    args = argparser()

    device = 'cuda:0'
    alpha = args.dirichlet_alpha
    num_labels = 10
    num_clients = args.num_clients
    num_rounds = args.num_rounds

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    writer = SummaryWriter()

    # datasets
    data_PATH = os.path.join(args.data_dir, 'cifar-10-batches-py')
    train_data, train_labels, test_data, test_labels = cifar10_dict(data_PATH)

    train_data = np.reshape(train_data, (-1,32,32,3))
    train_data = apply_transform(train_data, transform)
    test_data = np.reshape(test_data, (-1,32,32,3))
    test_data = apply_transform(test_data, transform)
    
    train_labels = torch.Tensor(train_labels).type(torch.long)
    test_labels = torch.Tensor(test_labels).type(torch.long)

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
    """
    for epoch in tqdm(range(200)):
        train_epoch(model, train_loader, args)
    """

if __name__=='__main__':
    main()