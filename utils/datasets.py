import os

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


from utils.parser import cifar10_dict

def get_partition(args, label_idx, data_size, num_labels):
    partition = []

    num_clients = args.num_clients

    if args.client_distribution == 'uniform':
        client_size = [int(data_size / num_clients)] * num_clients
    elif args.client_distribution == 'Dirichlet':
        client_size = np.random.dirichlet(np.ones(num_clients)*args.divide_dirichlet, size=1)
        client_size = np.squeeze(client_size)
    else:
        raise Exception('Wrong client distribution method')        

    # FIXIT: how to distribute non-iid?
    if args.divide_method == 'Dirichlet':
        client_dist = np.random.dirichlet([args.dirichlet_alpha] * 1.0 * num_labels, 
            size=num_clients)
    elif args.divide_method == 'uniform':
        client_dist = np.full((num_clients, num_labels), 1.0 / num_labels)
    else:
        raise Exception('Wrong divide method') 

    for client_id in range(num_clients):
        sample_idx = []
        size = client_size[client_id]
        sample_dist = client_dist[client_id]

        for label in range(num_labels):
            label_size = int(sample_dist[label] * size)
            sample = np.random.choice(label_idx[label], label_size).tolist()
            sample_idx += sample

        sample_idx = np.array(sample_idx)
        partition.append(sample_idx)

    assert len(partition) == num_clients
    return partition


def get_dataset(args):
    """
    Output
    - train_dataset: (train_data, train_labels)
    - test_dataset: (test_data, test_labels)
    - partition
    """
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if args.dataset == 'cifar10':
        num_labels = 10
        in_channel = 3
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)    
    elif args.dataset == 'cifar100':
        num_labels = 100
        in_channel = 3
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
    elif args.dataset == 'fmnist':
        num_labels = 10
        in_channel = 1
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=transform)
    else:
        raise Exception('Wrong dataset')

    train_labels = train_dataset.targets
    train_size = len(train_labels)

    label_idx = []
    for label in range(num_labels):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx += [idx]

    partition = get_partition(args, label_idx, train_size, num_labels)

    return train_dataset, test_dataset, partition, num_labels, in_channel