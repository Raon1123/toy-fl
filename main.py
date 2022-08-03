import cProfile, pstats
from pkgutil import get_data

import os
from datetime import datetime
import pickle, csv

import numpy as np
import torch
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


def main(args):
    device = get_device(args)
    experiment = exp_str(args)

    print("=" * 20)
    print("Experiment: ", experiment)
    print("=" * 20)
    
    alpha = args.dirichlet_alpha
    num_labels = 10
    num_clients = args.num_clients

    num_rounds = args.num_rounds
    writer = None

    log_PATH = os.path.join(args.logdir, experiment)
    writer = SummaryWriter(log_dir=log_PATH)

    os.makedirs(args.save_path, exist_ok=True)
    save_DIR = os.path.join(args.save_path, experiment)
    os.makedirs(save_DIR, exist_ok=True)

    # datasets
    train_dataset, test_dataset, partition = get_dataset(args) 

    train_data, train_labels = train_dataset
    test_data, test_labels = test_dataset

    train_loader = DataLoader(TensorDataset(train_data, train_labels), 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # make partition
    client_datasets = []
    active_client_bin = [0] * num_clients
    
    label_idx = []
    for label in range(num_labels):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx += [idx]

    # make tensor dataset
    for client_partition in partition:
        client_data = train_data[client_partition][:]
        client_label = train_labels[client_partition]
        client_dataset = TensorDataset(client_data, client_label)
        client_datasets.append(client_dataset)

    partition_PATH = os.path.join(log_PATH, args.dataset + "_partiton.pickle")
    with open(partition_PATH, "wb") as fw:
        pickle.dump(partition, fw)

    # hyperparam
    if args.model == 'CNN':
        model = CNN()
    elif args.model == 'ResNet18':
        model = resnet18(num_classes=num_labels)
    model = model.to(device)

    loss_array = None

    # train phase
    pbar = tqdm(range(num_rounds), desc='FL round')
    for round in pbar:
        active_idx, loss_array = run_round(model, client_datasets, args, loss_array)

        train_loss = np.sum(loss_array)

        # bin count
        for idx in active_idx:
            active_client_bin[idx] = active_client_bin[idx] + 1

        # test phase
        acc, _, avg_loss = test_epoch(model, test_loader, device)

        desc = {
            'Test Acc': acc,
            'Average Loss': avg_loss,
            'Train Loss': train_loss,
        }
        pbar.set_postfix(desc)

        # logging
        if (writer is not None) and ((round + 1) % args.log_freq == 0):
            prefix = 'FL'
            writer.add_scalar(prefix+'/Test Acc', acc, round)
            writer.add_scalar(prefix+'/Test Loss', avg_loss, round)
            writer.add_scalar(prefix+'/Train Loss', train_loss, round)
            save_loss(loss_array, round, save_DIR)

    writer.flush()

    log_bin(active_client_bin, partition, save_DIR)
    if args.model_save:
        save_model(model, save_DIR) 
    

def central_main(args):
    device = get_device(args)
    experiment = exp_str(args)

    log_PATH = os.path.join(args.logdir, experiment)
    writer = SummaryWriter(log_dir=log_PATH)

    os.makedirs(args.save_path, exist_ok=True)
    save_DIR = os.path.join(args.save_path, experiment)
    os.makedirs(save_DIR, exist_ok=True)

    train_dataset, test_dataset, _, num_labels = get_dataset(args)

    train_loader = DataLoader(train_dataset, 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("=== Centralized Setting ===")
    
    if args.model == 'CNN':
        center_model = CNN()
    elif args.model == 'ResNet18':
        center_model = resnet18(num_classes=num_labels)
    center_model = center_model.to(device)

    acc = 0
    test_loss = 0.0

    pbar = tqdm(range(args.central_epoch))

    for epoch in pbar:
        train_loss, test_loss = 0.0, 0.0
        train_loss = train_epoch(center_model, train_loader, args, device)
        
        if (writer is not None) and ((epoch + 1) % args.log_freq == 0):
            acc, test_loss = test_epoch(center_model, test_loader, device)
            prefix = 'Centralized'
            writer.add_scalar(prefix+'/Test Acc', acc, epoch)
            writer.add_scalar(prefix+'/Test Loss', test_loss, epoch)
            writer.add_scalar(prefix+'/Train Loss', train_loss, epoch)
            pbar.set_postfix({'Acc': acc, 'Test loss': test_loss, 'Train loss': train_loss})
    writer.flush()


if __name__=='__main__':
    args = argparser()

    if not args.centralized:
        main(args)
    
    central_main(args)