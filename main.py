import cProfile, pstats

import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from models.cnn import CNN
from utils.parser import cifar10_dict, argparser
from utils.epochs import test_epoch, run_round

def exp_str(args):
    join_list = []

    model_str = 'CNN'
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


@profile
def main(args):
    device = args.device
    experiment = exp_str(args)
    
    alpha = args.dirichlet_alpha
    num_labels = 10
    num_clients = args.num_clients

    num_rounds = args.num_rounds
    writer = None

    log_PATH = os.path.join(args.logdir, experiment)
    writer = SummaryWriter(log_dir=log_PATH)

    # datasets
    data_PATH = os.path.join(args.data_dir, 'cifar-10-batches-py')
    train_data, train_labels, test_data, test_labels = cifar10_dict(data_PATH)    

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_sz = len(train_labels)

    # make partition
    partition = []
    client_datasets = []
    client_size = np.random.dirichlet([alpha] * num_clients, size=1) # life is RANDOM
    client_dist = np.random.dirichlet([alpha] * 10, size=num_clients) # distribution of client
    
    label_idx = []
    for label in range(num_labels):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx += [idx]

    # sampling
    for client_id in range(num_clients):
        sample_idx = []
        size = int(train_sz * client_size[0, client_id]) + 1
        sample_dist = client_dist[client_id]

        for label in range(num_labels):
            sample = int(sample_dist[label] * size)
            sample = np.random.choice(label_idx[label], size).tolist()
            sample_idx += sample

        sample_idx = np.array(sample_idx)
        partition.append(sample_idx)

        client_data = train_data[sample_idx, :]
        client_label = train_labels[sample_idx]
        client_loader = TensorDataset(client_data, client_label)
        client_datasets.append(client_loader)

    # hyperparam
    model = CNN().to(device)

    # train phase
    pbar = tqdm(range(num_rounds), desc='FL round')
    for round in pbar:
        run_round(model, client_datasets, args)

        # test phase
        acc, _, avg_loss = test_epoch(model, test_loader, device)

        if (writer is not None) and (round % args.log_freq == 0):
            writer.add_scalar('Test Acc', acc, round)
            writer.add_scalar('Test Loss', avg_loss, round)

    if args.model_save:
        os.makedirs(args.save_path, exist_ok=True)
        save_PATH = os.path.join(args.save_path, experiment) + '.pt'
        torch.save(model.state_dict(), save_PATH)

    print("=== Centralized ===")
    """
    center_model = CNN().to(device)
    for epoch in tqdm(range(200)):
        train_epoch(center_model, train_loader, args, device)
    acc, loss, _ = test_epoch(center_model, test_loader, device)
    print(acc)
    """

if __name__=='__main__':
    args = argparser()

    if args.profile:
        profiler = cProfile.Profile()
        
        profiler.enable()
        main(args)
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
    else:
        main(args)