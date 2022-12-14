import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from models.modelutil import get_model
from utils.parser import argparser, get_device
from utils.epochs import test_epoch, run_round
import utils.logger as logger
from utils.datasets import get_dataset
from utils.toolkit import set_seed


def main(args, writer, seed):
    device = get_device(args)
    
    logger.print_experiment(args)
    save_DIR, loss_DIR = logger.get_save_dir(args, seed)
    
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    acc_list = []

    # datasets
    train_dataset, test_dataset, partition, num_classes, in_channel = get_dataset(args, seed) 

    partition_weight = np.array([len(l) for l in partition]) 
    partition_weight = partition_weight / np.sum(partition_weight) # normalzie

    test_loader = DataLoader(test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers)

    active_client_bin = [0] * num_clients

    model = get_model(args, num_classes, in_channel)
    print(model)
    model = model.to(device)

    loss_array = None
    client_params = None

    # train phase
    pbar = tqdm(range(num_rounds), desc='FL round')
    for communication_round in pbar:
        ret = run_round(model, 
            train_dataset, 
            partition, 
            args, 
            prev_losses=loss_array, 
            prev_params=client_params)
        active_idx, loss_array, client_params = ret

        if args.verbose:
            print("Round ", communication_round)
            print("Selected client", active_idx)
            print("Loss array", loss_array)
        train_loss = np.sum(loss_array * partition_weight)
            
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
        acc_list.append(acc)

        # logging
        if (writer is not None) and ((communication_round + 1) % args.log_freq == 0):
            prefix = 'FL'
            writer.add_scalar(prefix+'/Test Acc', acc, communication_round)
            writer.add_scalar(prefix+'/Test Loss', test_loss, communication_round)
            writer.add_scalar(prefix+'/Train Loss', train_loss, communication_round)
            logger.save_loss(loss_array, communication_round, loss_DIR)

    writer.flush()

    logger.save_bin(active_client_bin, partition, save_DIR, seed)
    logger.save_acc(acc_list, save_DIR, seed)
    if args.model_save:
        logger.save_model(model, save_DIR, seed) 


if __name__=='__main__':
    logger.write_timestamp("Start")
    args = argparser()

    seed_list = args.seeds

    for seed in seed_list:
        print("seed", seed)
        set_seed(seed)

        writer = logger.get_writer(args, seed)

        if not args.centralized:
            main(args, writer, seed)
        logger.write_timestamp("End FL")
    