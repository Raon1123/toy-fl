import os
import pickle
import argparse

import numpy as np
from scipy import io

import torch
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms

from utils.consts import *
from utils.toolkit import get_dataset_labels
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
    print("Label distribution", args.label_distribution)
    print("=" * 30)

    class_size = np.zeros(num_classes)
    for idx in range(num_classes):
        class_size[idx] = len(label_idx[idx])

    if args.label_distribution == 'Dirichlet':
        label_dist = np.random.dirichlet([args.label_dirichlet] * num_clients, 
            size=num_classes) 
        label_dist = label_dist.T # (num_clients, num_classes)
    elif args.label_distribution == 'uniform':
        label_dist = np.full((num_clients, num_classes), 1.0 / num_clients)
    elif args.label_distribution == 'random':
        candidate_data_index = []
        for labels in label_idx:
            candidate_data_index = candidate_data_index + labels  
    else:
        raise Exception('Wrong divide method') 

    client_dist = label_dist * class_size 
    client_dist = np.trunc(client_dist).astype(int) # int array (num_clients, num_classes)

    for client_id in range(num_clients):
        sample_idx = []
        distribution = client_dist[client_id,:] # (num_classes,)

        if args.label_distribution == 'random':
            sample_idx = np.random.choice(candidate_data_index, np.sum(distribution))
            for idx in sample_idx:
                candidate_data_index.remove(idx)
        else:
            for label in range(num_classes):
                sample_size = min(distribution[label], len(label_idx[label]))
                if len(label_idx[label]) == 0:
                    continue
                sample = np.random.choice(label_idx[label], sample_size, replace=False)
                label_idx[label] = np.setdiff1d(label_idx[label], sample)
                sample_idx += sample.tolist()

        partition.append(sample_idx)

    # for remain part
    # randomly add
    for label in range(num_classes):
        remain = label_idx[label]
        for idx in remain:
            client_idx = np.random.choice(num_clients, 1).item()
            partition[client_idx].append(idx)

    partition = [np.array(l) for l in partition]

    assert len(partition) == num_clients

    return partition


def get_dataset(args, seed):
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
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                        download=True, transform=transform)    
    elif args.dataset == 'cifar100':
        num_classes = 100
        in_channel = 3
        train_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False,
                                        download=True, transform=transform)
    elif args.dataset == 'fmnist':
        num_classes = 10
        in_channel = 1
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
        train_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir, train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir, train=False,
                                        download=True, transform=transform)
    elif args.dataset == 'cifar10feature':
        num_classes = 10
        in_channel = 1
        train_dataset, test_dataset = get_cifar10feature(args)
    else:
        raise Exception('Wrong dataset')

    train_labels = get_dataset_labels(train_dataset)
    train_size = len(train_labels)

    label_idx = []
    for label in range(num_classes):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx.append(idx)

    join_list = [args.dataset, args.label_distribution, str(args.num_clients), str(seed), "partiton.pickle"]
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


def get_cifar10feature(args):
    data_dir = args.data_dir

    # train dataset
    trn_name = 'CIFAR10Trn.mat'
    mat_path = os.path.join(data_dir, trn_name)
    mat = io.loadmat(mat_path)
    trnX = torch.Tensor(mat['Trn'][0][0][0])
    trnY = torch.Tensor(mat['Trn'][0][0][1]).argmax(axis=1)

    # test dataset
    tst_name = 'CIFAR10Tst.mat'
    mat_path = os.path.join(data_dir, tst_name)
    mat = io.loadmat(mat_path)
    tstX = torch.Tensor(mat['Tst'][0][0][0])
    tstY = torch.Tensor(mat['Tst'][0][0][1]).argmax(axis=1)

    train_dataset = TensorDataset(trnX, trnY)
    test_dataset = TensorDataset(tstX, tstY)

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data',
        help='root directory of data')
    parser.add_argument('--logdir', type=str, default='../logdir')
    parser.add_argument('--dataset', type=str, default='cifar10',
        choices=DATASET,
        help='experiment dataset')

    parser.add_argument('-C', '--num_clients', type=int, default=100)
    parser.add_argument('--client_distribution', type=str, default='uniform',
        choices=['Dirichlet', 'IID'],
        help='distribution of number of clients')
    parser.add_argument('--client_dirichlet', type=float, default=10)

    parser.add_argument('--label_distribution', type=str, default='Dirichlet',
        choices=['Dirichlet', 'random', 'uniform'],
        help='distribution of labels in client')
    parser.add_argument('--label_dirichlet', type=float, default=0.2,
        help='divide method')

    args = parser.parse_args()
    _, _, partition, _, _ = get_dataset(args)
    print("Partition Generation Done")
    summation = 0
    for part in partition:
        summation += len(part)
        print(len(part))

    print(summation)