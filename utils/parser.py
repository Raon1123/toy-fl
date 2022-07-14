import os
import pickle
import argparse

import torch
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cifar10_dict(PATH):
    """
    Parse CIFAR10 datasets
    
    Outputs
    - train_data
    - train_labels
    - test_data
    - test_labels
    """
    # parse train batch
    batches = 5
    train_data, train_labels = None, None
    
    for batch in range(1, batches+1):
        file_name = 'data_batch_' + str(batch)
        file_path = os.path.join(PATH, file_name)

        batch_dict = unpickle(file_path)
        data = batch_dict[b'data']
        labels = batch_dict[b'labels']

        if train_data is None:
            train_data = data
            train_labels = labels
        else:
            train_data = np.concatenate((train_data, data))
            train_labels = np.concatenate((train_labels, labels))
    
    # parse test batch
    file_name = 'test_batch'
    file_path = os.path.join(PATH, file_name)
    batch_dict = unpickle(file_path)
    test_data = batch_dict[b'data']
    test_labels = batch_dict[b'labels']

    return train_data, train_labels, test_data, test_labels


def argparser():
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--data_dir', type=str, default='./data',
        help='root directory of data')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.2)

    parser.add_argument('-C', '--num_clients', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-A', '--active_selection', type=int, default=10)
    parser.add_argument('-R', '--num_rounds', type=int, default=2000)

    # local hyperparameter
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=5e-4,
        help='local learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='local momentum for SGD')
    parser.add_argument('--local_epoch', type=int, default=1)

    args = parser.parse_args()
    return args


def apply_transform(imgs, transform):
    size = imgs.shape[0]
    results = torch.zeros((size, 3, 32, 32))

    for idx in range(size):
        img = transform(imgs[idx])
        results[idx, :] = img

    return results