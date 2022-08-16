import cProfile, pstats
from pkgutil import get_data

import os
from datetime import datetime
import pickle, csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from models.cnn import CNN
from utils.parser import argparser, get_device
from utils.epochs import test_epoch, train_epoch, run_round
from utils.logger import log_bin, save_model, save_loss
from utils.datasets import get_dataset

def exp_str(args):
    join_list = []

    model_str = args.model
    join_list.append(model_str)

    active_str = args.active_algorithm
    join_list.append(active_str)

    dirichlet_str = 'CIFAR10_' + str(args.dirichlet_alpha)
    join_list.append(dirichlet_str)

    now = datetime.now()
    now_str = now.strftime('%y%m%d-%H%M%S')
    join_list.append(now_str)

    ret = '_'.join(join_list)
    return ret


def main(args, writer):
    device = get_device(args)
    experiment = exp_str(args)

    print("=" * 20)
    print("Experiment: ", experiment)
    print("=" * 20)
    
    num_clients = args.num_clients
    num_rounds = args.num_rounds

    os.makedirs(args.save_path, exist_ok=True)
    save_DIR = os.path.join(args.save_path, experiment)
    os.makedirs(save_DIR, exist_ok=True)

    # datasets
    train_dataset, test_dataset, partition, num_classes, in_channel = get_dataset(args) 

    test_loader = DataLoader(test_dataset, 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    active_client_bin = [0] * num_clients

    partition_PATH = os.path.join(log_PATH, args.dataset + "_partiton.pickle")
    with open(partition_PATH, "wb") as fw:
        pickle.dump(partition, fw)

    # hyperparam
    if args.model == 'CNN':
        model = CNN(in_channel=in_channel, num_classes=num_classes)
    elif args.model == 'ResNet18':
        model = resnet18(num_classes=num_classes)
    model = model.to(device)

    loss_array = None

    # train phase
    pbar = tqdm(range(num_rounds), desc='FL round')
    for round in pbar:
        active_idx, loss_array = run_round(model, train_dataset, partition, args, loss_array)

        train_loss = np.sum(loss_array)

        # bin count
        for idx in active_idx:
            active_client_bin[idx] = active_client_bin[idx] + 1

        # test phase
        acc, test_loss = test_epoch(model, test_loader, device)

        desc = {
            'Test Acc': acc,
            'Test Loss': test_loss,
            'Train Loss': train_loss
        }
        pbar.set_postfix(desc)

        # logging
        if (writer is not None) and ((round + 1) % args.log_freq == 0):
            prefix = 'FL'
            writer.add_scalar(prefix+'/Test Acc', acc, round)
            writer.add_scalar(prefix+'/Test Loss', test_loss, round)
            writer.add_scalar(prefix+'/Train Loss', train_loss, round)
            save_loss(loss_array, round, save_DIR)

    writer.flush()

    log_bin(active_client_bin, partition, save_DIR)
    if args.model_save:
        save_model(model, save_DIR) 
    

def central_main(args, writer):
    device = get_device(args)
    experiment = exp_str(args)

    os.makedirs(args.save_path, exist_ok=True)
    save_DIR = os.path.join(args.save_path, experiment)
    os.makedirs(save_DIR, exist_ok=True)

    train_dataset, test_dataset, _, num_classes, in_channel = get_dataset(args)

    train_loader = DataLoader(train_dataset, 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("=== Centralized Setting ===")
    
    if args.model == 'CNN':
        center_model = CNN(num_classes=num_classes, in_channel=in_channel)
    elif args.model == 'ResNet18':
        center_model = resnet18(num_classes=num_classes, in_channel=in_channel)
    center_model = center_model.to(device)
    optimizer = optim.SGD(center_model.parameters(), lr=args.lr, momentum=args.momentum)
    lossf = nn.CrossEntropyLoss()

    acc = 0
    test_loss = 0.0

    pbar = tqdm(range(args.central_epoch))

    for epoch in pbar:
        train_loss, test_loss = 0.0, 0.0
        train_loss = train_epoch(center_model, optimizer, lossf, train_loader, args, device)
        
        if (writer is not None) and ((epoch + 1) % args.log_freq == 0):
            acc, test_loss = test_epoch(center_model, test_loader, device)
            prefix = 'Centralized'
            writer.add_scalar(prefix+'/Test Acc', acc, epoch)
            writer.add_scalar(prefix+'/Test Loss', test_loss, epoch)
            writer.add_scalar(prefix+'/Train Loss', train_loss, epoch)
            pbar.set_postfix({'Acc': acc, 'Test loss': test_loss, 'Train loss': train_loss})
    writer.flush()


def write_timestamp(prefix=""):
    now = datetime.now()
    now_str = now.strftime('%y%m%d-%H%M%S')
    print(prefix, now_str)


if __name__=='__main__':
    write_timestamp("Start")
    args = argparser()

    experiment = exp_str(args)
    log_PATH = os.path.join(args.logdir, experiment)
    writer = SummaryWriter(log_dir=log_PATH)

    if not args.centralized:
        main(args, writer)
    write_timestamp("End FL")
    
    central_main(args, writer)
    write_timestamp("End CL")