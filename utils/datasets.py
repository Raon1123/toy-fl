import os
import pickle
import argparse

import numpy as np
import torchvision
import torchvision.transforms as transforms

#from utils.parser import argparser

def get_partition(args, label_idx, num_classes, data_size):
    """
    Generate client partition
    - label_index(list; num_classes,): list index of label
    - num_classes (int): number of classes in datasets
    - data_size (int): total size of dataset
    """
    partition = []

    num_clients = args.num_clients

    # description
    print("=" * 30)
    print("Number of clients", num_clients)
    print("Client distribution", args.client_distribution)
    print("Label distribution", args.label_distribution)
    print("=" * 30)

    if args.client_distribution == 'uniform':
        client_size = [int(data_size / num_clients)] * num_clients
    elif args.client_distribution == 'Dirichlet':
        client_size = np.random.dirichlet(np.ones(num_clients)*args.client_dirichlet, size=1)
        client_size = np.squeeze(client_size)
    else:
        raise Exception('Wrong client distribution method')        

    if args.label_distribution == 'Dirichlet':
        label_dist = np.random.dirichlet([args.label_dirichlet] * num_classes, 
            size=num_clients)
    elif args.label_distribution == 'uniform':
        label_dist = np.full((num_clients, num_classes), 1.0 / num_classes)
    elif args.label_distribution == 'random':
        candidate_data_index = []
        for labels in label_idx:
            candidate_data_index = candidate_data_index + labels  
    else:
        raise Exception('Wrong divide method') 

    for client_id in range(num_clients):
        sample_idx = []
        size = client_size[client_id]

        if args.label_distribution == 'random':
            sample_idx = np.random.choice(candidate_data_index, size)
            for idx in sample_idx:
                candidate_data_index.remove(idx)
        else:
            sample_dist = label_dist[client_id]
            for label in range(num_classes):
                label_size = int(sample_dist[label] * size)
                if len(label_idx[label]) == 0:
                    continue
                sample = np.random.choice(label_idx[label], label_size)
                label_idx[label] = np.setdiff1d(label_idx[label], sample)
                sample_idx += sample.tolist()
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
        num_classes = 10
        in_channel = 3
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)    
    elif args.dataset == 'cifar100':
        num_classes = 100
        in_channel = 3
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
    elif args.dataset == 'fmnist':
        num_classes = 10
        in_channel = 1
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=transform)
    else:
        raise Exception('Wrong dataset')

    train_labels = train_dataset.targets
    train_size = len(train_labels)

    label_idx = []
    for label in range(num_classes):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx.append(idx)

    join_list = [args.dataset, args.label_distribution, str(args.num_clients), "partiton.pickle"]
    partition_file = '_'.join(join_list)
    partition_PATH = os.path.join(args.logdir, partition_file)
    if os.path.exists(partition_PATH):
        print("Load partition")
        with open(partition_PATH, "rb") as fr:
            partition = pickle.load(fr)
    else:
        print("Generate partition")
        partition = get_partition(args, label_idx, num_classes, train_size)
        with open(partition_PATH, "wb") as fw:
            pickle.dump(partition, fw)
    
    return train_dataset, test_dataset, partition, num_classes, in_channel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data',
        help='root directory of data')
    parser.add_argument('--logdir', type=str, default='./logdir')
    parser.add_argument('--dataset', type=str, default='cifar10',
        choices=['cifar10', 'cifar100', 'fmnist'],
        help='experiment dataset')

    parser.add_argument('-C', '--num_clients', type=int, default=100)
    parser.add_argument('--client_distribution', type=str, default='uniform',
        choices=['Dirichlet', 'IID'],
        help='distribution of number of clients')
    parser.add_argument('--client_dirichlet', type=float, default=10)

    parser.add_argument('--label_distribution', type=str, default='uniform',
        choices=['Dirichlet', 'random', 'uniform'],
        help='distribution of labels in client')
    parser.add_argument('--label_dirichlet', type=float, default=0.2,
        help='divide method')

    args = parser.parse_args()
    _ = get_dataset(args)
    print("Partition Generation Done")