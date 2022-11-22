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
from utils.epochs import test_epoch, run_round
from utils.logger import save_bin, save_model, save_loss, save_acc
from utils.logger import exp_str, write_timestamp
from utils.datasets import get_dataset
from utils.toolkit import set_seed


def main(args, writer, seed):
    device = get_device(args)
    experiment = exp_str(args)

    print("=" * 20)
    print("Experiment: ", experiment)
    print("=" * 20)
    
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    acc_list = []

    os.makedirs(args.save_path, exist_ok=True)

    save_DIR = os.path.join(args.save_path, experiment)
    os.makedirs(save_DIR, exist_ok=True)

    loss_DIR = os.path.join(save_DIR, 'loss_'+str(seed))
    os.makedirs(loss_DIR, exist_ok=True)

    # datasets
    train_dataset, test_dataset, partition, num_classes, in_channel = get_dataset(args, seed) 

    test_loader = DataLoader(test_dataset, 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
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
        train_loss = np.average(loss_array)
            
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
            save_loss(loss_array, communication_round, loss_DIR)

    writer.flush()

    save_bin(active_client_bin, partition, save_DIR, seed)
    save_acc(acc_list, save_DIR, seed)
    if args.model_save:
        save_model(model, save_DIR, seed) 


if __name__=='__main__':
    write_timestamp("Start")
    args = argparser()

    seed_list = args.seeds

    for seed in seed_list:
        print("seed", seed)
        set_seed(seed)

        experiment = exp_str(args)
        log_PATH = os.path.join(args.logdir, experiment)
        log_PATH = os.path.join(log_PATH, str(seed))
        writer = SummaryWriter(log_dir=log_PATH)

        if not args.centralized:
            main(args, writer, seed)
        write_timestamp("End FL")
    