import os
import pickle
import argparse

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

from utils.consts import *

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_device(args):
    device = None
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda:' + args.device

    assert device != None
    return device


def get_optimizer(model, args):
    optimizer = None

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError

    assert optimizer is not None
    return optimizer


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

    # FL settings
    parser.add_argument('-N', '--num_clients', type=int, default=100)
    parser.add_argument('-R', '--num_rounds', type=int, default=200)
    parser.add_argument('--model', type=str, default='CNN',
        choices=MODELS)
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--label_distribution', type=str, default='uniform',
        choices=['Dirichlet', 'random', 'uniform', 'shard'],
        help='distribution of labels in client')
    parser.add_argument('--label_dirichlet', type=float, default=0.2,
        help='divide method')
    parser.add_argument('--shard_per_client', type=int, default=1,
        help='shard_per_client')

    # active client selection settings
    parser.add_argument('-C', '--active_selection', type=int, default=5)
    parser.add_argument('--active_algorithm', type=str, default='Random',
        choices=ACTIVEALGORITHM,
        help='Active client selection strategy')

    # pow-d
    parser.add_argument('--powd', type=int, default=10,
        help='d value of power-of-d method')

    # FedCor
    parser.add_argument('--warmup',type = int, default=15,
                        help = 'length of warm up phase for GP')
    parser.add_argument('--gpr_begin', type = int,default=10,
                        help='the round begin to sample and train GP')
    parser.add_argument('--GPR_interval', type = int, default=10, 
                        help='interval of sampling and training of GP, namely, Delta t')
    parser.add_argument('--group_size',type = int, default = 11, 
                        help='length of history round to sample for GP, equal to M Delta t + 1 in paper')
    parser.add_argument('--GPR_gamma',type = float,default = 0.8,
                        help='gamma for training GP')
    parser.add_argument('--GPR_Epoch',type=int,default=100,
                        help='number of optimization iterations of GP')
    parser.add_argument('--fedcor_beta', type=float, default=0.95,
                        help='beta for FedCor')

    # gradient based approach
    parser.add_argument('--similarity_measure', type=str, default='distance')

    # local hyperparametercd
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=5e-3,
        help='local learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='local momentum for SGD')
    parser.add_argument('--local_epoch', type=int, default=5)

    parser.add_argument('--logdir', type=str, default='./logdir')
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    
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
    parser.add_argument('--mlp_layers',type= int,default=[64,],nargs="+",
                        help="numbers of dimensions of each hidden layer in MLP, or fc layers in CNN")
    parser.add_argument('--depth',type = int,default = 20, 
                        help = "The depth of ResNet. Only valid when model is resnet")

    parser.add_argument('--seeds', nargs='+', type=int, default=[0])

    args = parser.parse_args()
    return args

