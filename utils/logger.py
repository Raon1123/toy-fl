import os
import csv
from datetime import datetime

import torch
import numpy as np
import pickle

from torch.utils.tensorboard import SummaryWriter

def write_timestamp(prefix=""):
    now = datetime.now()
    now_str = now.strftime('%y%m%d-%H%M%S')
    print(prefix, now_str)


def print_experiment(args):
    print("=" * 20)
    print("device: ", args.device)
    print("dataset: ", args.dataset)
    print("num_clients: ", args.num_clients)

    print("label_distribution: ", args.label_distribution)
    print("label_dirichlet: ", args.label_dirichlet)

    print("models: ", args.model)
    print("num_rounds: ", args.num_rounds)
    print("active_algorithm: ", args.active_algorithm)

    print("postfix: ", args.postfix)
    print("=" * 20)


def exp_str(args):
    join_list = [
        args.model,
        args.active_algorithm]

    # distribution settings
    join_list.append(args.label_distribution)
    if args.label_distribution == 'Dirichlet':
        join_list.append(str(args.label_dirichlet))

    if args.postfix != '':
        join_list.append(args.postfix)

    ret = '_'.join(join_list)
    return ret


def save_bin(bins, partition, bin_DIR, seed):
    bin_PATH = os.path.join(bin_DIR, 'bin_'+str(seed)+'.csv')

    bin_file = open(bin_PATH, "w")
    bin_writer = csv.writer(bin_file)

    for idx, bin in enumerate(bins):
        sz = len(partition[idx])
        row = [idx, bin, sz]
        bin_writer.writerow(row)
    bin_file.close()


def save_model(model, model_DIR, seed):
    save_PATH = os.path.join(model_DIR, 'save'+str(seed)+'.pt')
    torch.save(model.state_dict(), save_PATH)


def save_loss(loss_array, round, save_DIR):
    if type(round) is not str:
        round = str(round)

    loss_PATH = os.path.join(save_DIR, 'loss_' + round + '.npy')

    np.save(loss_PATH, loss_array)


def save_acc(acc_list, save_DIR, seed):
    acc_PATH = os.path.join(save_DIR, 'acc'+str(seed)+'.pkl')
    with open(acc_PATH, 'wb') as f:
        pickle.dump(acc_list, f)


def get_writer(args, seed=0):
    experiment = exp_str(args)
    log_PATH = os.path.join(args.logdir, args.dataset, experiment)
    log_PATH = os.path.join(log_PATH, str(seed))
    writer = SummaryWriter(log_dir=log_PATH)

    return writer


def get_save_dir(args, seed=0):
    experiment = exp_str(args)
    os.makedirs(args.save_path, exist_ok=True)

    save_DIR = os.path.join(args.save_path, args.dataset, experiment)
    os.makedirs(save_DIR, exist_ok=True)

    loss_DIR = os.path.join(save_DIR, 'loss_'+str(seed))
    os.makedirs(loss_DIR, exist_ok=True)

    return save_DIR, loss_DIR