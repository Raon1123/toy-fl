import os
import csv
from datetime import datetime

import torch
import numpy as np
import pickle

def write_timestamp(prefix=""):
    now = datetime.now()
    now_str = now.strftime('%y%m%d-%H%M%S')
    print(prefix, now_str)


def exp_str(args):
    join_list = [args.model,
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