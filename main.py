import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from models.modelutil import get_model
from utils.parser import argparser, get_device
from utils.epochs import test_epoch, train_epoch, run_round
from utils.logger import log_bin, save_model, save_loss
from utils.logger import exp_str, write_timestamp
from utils.datasets import get_dataset
from utils.toolkit import get_last_param


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

    model = get_model(args, num_classes, in_channel)
    print(model)
    model = model.to(device)

    loss_array = None
    param_list = None
    prev_model = None
    pseudograd = None

    # train phase
    pbar = tqdm(range(num_rounds), desc='FL round')
    for round in pbar:
        if prev_model is not None:
            prev_last = get_last_param(prev_model)
            curr_last = get_last_param(model)
            pseudograd = curr_last - prev_last
        prev_model = copy.deepcopy(model)

        ret = run_round(model, 
            train_dataset, 
            partition, 
            args, 
            prev_grad=pseudograd,
            prev_losses=loss_array, 
            prev_params=param_list)
        active_idx, loss_array, param_list = ret

        train_loss = np.sum(loss_array)

        print(active_idx)
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
    
    center_model = get_model(args, num_classes, in_channel)
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