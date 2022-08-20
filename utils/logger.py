import os
import csv
from datetime import datetime

import torch
import numpy as np

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


def log_bin(bins, partition, bin_DIR):
    os.makedirs(bin_DIR, exist_ok=True)
    bin_PATH = os.path.join(bin_DIR, 'bin.csv')

    bin_file = open(bin_PATH, "w")
    bin_writer = csv.writer(bin_file)

    for idx, bin in enumerate(bins):
        sz = len(partition[idx])
        row = [idx, bin, sz]
        bin_writer.writerow(row)
    bin_file.close()


def write_timestamp(prefix=""):
    now = datetime.now()
    now_str = now.strftime('%y%m%d-%H%M%S')
    print(prefix, now_str)


def save_model(model, model_DIR):
    save_PATH = os.path.join(model_DIR, 'save.pt')
    torch.save(model.state_dict(), save_PATH)


def save_loss(loss_array, round, save_DIR):
    if type(round) is not str:
        round = str(round)

    loss_PATH = os.path.join(save_DIR, 'loss_' + round + '.npy')

    np.save(loss_PATH, loss_array)