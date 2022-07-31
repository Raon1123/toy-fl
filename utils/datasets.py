import os

import numpy as np
from torch.utils.data import TensorDataset

from utils.parser import cifar10_dict

def get_partition(args, label_idx, data_size, num_labels):
    partition = []

    num_clients = args.num_clients

    if args.divide_method == 'uniform':
        client_size = [int(data_size / num_clients)] * num_clients
    else:
        raise Exception('Wrong divide method')        

    # FIXIT: how to distribute non-iid?
    client_dist = np.random.dirichlet([args.dirichlet_alpha] * 10, size=num_clients)

    for client_id in range(num_clients):
        sample_idx = []
        size = client_size[client_id]
        sample_dist = client_dist[client_id]

        for label in range(num_labels):
            sample = int(sample_dist[label] * size)
            sample = np.random.choice(label_idx[label], size).tolist()
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
    if args.dataset == 'cifar10':
        num_labels = 10
        data_PATH = os.path.join(args.data_dir, 'cifar-10-batches-py')
        train_data, train_labels, test_data, test_labels = cifar10_dict(data_PATH)    
    else:
        raise Exception('Wrong dataset')

    train_size = len(train_labels)

    label_idx = []
    for label in range(num_labels):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx += [idx]

    partition = get_partition(args, label_idx, train_size, num_labels)

    train_dataset = (train_data, train_labels)
    test_dataset = (test_data, test_labels)

    return train_dataset, test_dataset, partition