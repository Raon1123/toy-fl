import os
import pickle
import argparse

import torch
import torchvision.transforms as transforms
import numpy as np

from utils.consts import *

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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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

    train_data = np.reshape(train_data, (-1,32,32,3))
    train_data = apply_transform(train_data, transform)
    test_data = np.reshape(test_data, (-1,32,32,3))
    test_data = apply_transform(test_data, transform)

    train_labels = torch.Tensor(train_labels).type(torch.long)
    test_labels = torch.Tensor(test_labels).type(torch.long)

    return train_data, train_labels, test_data, test_labels


def get_device(args):
    device = None
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda:' + args.device

    assert device != None
    return device


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--postfix', type=str, default='', help='postfix of experiment')

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--device', type=str, default='0',
        help='cuda device number')

    # directory
    parser.add_argument('--data_dir', type=str, default='./data',
        help='root directory of data')
    parser.add_argument('--dataset', type=str, default='cifar10',
        choices=DATASET,
        help='experiment dataset')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory',  action='store_true')

    parser.add_argument('-C', '--num_clients', type=int, default=100)
    parser.add_argument('-R', '--num_rounds', type=int, default=2000)
    parser.add_argument('--model', type=str, default='CNN',
        choices=['ResNet18', 'CNN', 'NaiveCNN'])
    parser.add_argument('--batch_size', type=int, default=50)
    
    parser.add_argument('--client_distribution', type=str, default='uniform',
        choices=['Dirichlet', 'IID'],
        help='distribution of number of clients')
    parser.add_argument('--client_dirichlet', type=float, default=10.)

    parser.add_argument('--label_distribution', type=str, default='uniform',
        choices=['Dirichlet', 'random', 'uniform'],
        help='distribution of labels in client')
    parser.add_argument('--label_dirichlet', type=float, default=0.2,
        help='divide method')

    parser.add_argument('-A', '--active_selection', type=int, default=10)
    parser.add_argument('--active_algorithm', type=str, default='Random',
        choices=['Random', 'LossSampling'],
        help='Active client selection strategy')

    # local hyperparametercd
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=5e-3,
        help='local learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='local momentum for SGD')
    parser.add_argument('--local_epoch', type=int, default=5)

    parser.add_argument('--logdir', type=str, default='./logdir')
    parser.add_argument('--log_freq', type=int, default=10)
    
    parser.add_argument('--model_save', action='store_false')
    parser.add_argument('--save_path', type=str, default='./save')

    # Centralized setting
    parser.add_argument('--centralized', action='store_true')
    parser.add_argument('--central_epoch', type=int, default=500)

    parser.add_argument('--kernel_sizes', type=int, default=[3, 3, 3], nargs="*",
                        help='kernel size in each convolutional layer')
    parser.add_argument('--num_filters', type=int, default=[32, 64, 64], nargs = "*",
                        help="number of filters in each convolutional layer.")
    parser.add_argument('--padding', action='store_true', 
                        help='use padding in each convolutional layer')
    parser.add_argument('--mlp_layers',type= int,default=[64,],nargs="*",
                        help="numbers of dimensions of each hidden layer in MLP, or fc layers in CNN")
    parser.add_argument('--depth',type = int,default = 20, 
                        help = "The depth of ResNet. Only valid when model is resnet")

    args = parser.parse_args()
    return args


def apply_transform(imgs, transform):
    size = imgs.shape[0]
    results = torch.zeros((size, 3, 32, 32))

    for idx in range(size):
        img = transform(imgs[idx])
        results[idx, :] = img

    return results