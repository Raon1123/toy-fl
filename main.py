import cProfile, pstats

import os
from datetime import datetime
import pickle, csv

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from models.cnn import CNN
from utils.parser import cifar10_dict, argparser, get_device
from utils.epochs import test_epoch, train_epoch, run_round
from utils.logger import log_bin, save_model

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


def main(args):
    device = get_device(args)
    experiment = exp_str(args)

    print("=" * 10)
    print("Experiment: ", experiment)
    print("=" * 10)
    
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
    active_client_bin = [0] * num_clients

    client_size = None
    client_dist = None
    
    if args.divide_method == 'Dirichlet':
        client_size = np.random.dirichlet([alpha] * num_clients, size=1) # life is RANDOM
        client_size = np.squeeze(client_size)
        client_size = (train_sz * client_size).astype(int) + 1
    elif args.divide_method == 'Samesize':
        client_size = [int(train_sz / num_clients)] * num_clients

    client_dist = np.random.dirichlet([alpha] * 10, size=num_clients) # distribution of client
    
    assert client_size is not None
    assert client_dist is not None
    
    label_idx = []
    for label in range(num_labels):
        idx = np.where(np.array(train_labels) == label)[0]
        label_idx += [idx]

    # sampling
    for client_id in range(num_clients):
        sample_idx = []
        size = client_size[client_id]
        sample_dist = client_dist[client_id]

        for label in range(num_labels):
            sample = int(sample_dist[label] * size)
            sample = np.random.choice(label_idx[label], size).tolist()
            sample_idx += sample

        sample_idx = np.array(sample_idx)
        partition.append(sample_idx)

        client_data = train_data[sample_idx][:]
        client_label = train_labels[sample_idx]
        client_dataset = TensorDataset(client_data, client_label)
        client_datasets.append(client_dataset)

    partition_PATH = os.path.join(log_PATH, "partiton.pickle")
    with open(partition_PATH, "wb") as fw:
        pickle.dump(partition, fw)

    # hyperparam
    model = CNN().to(device)

    # train phase
    pbar = tqdm(range(num_rounds), desc='FL round')
    for round in pbar:
        active_idx, train_loss = run_round(model, client_datasets, args)

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
        if (writer is not None) and (round % args.log_freq == 0):
            writer.add_scalar('Test Acc', acc, round)
            writer.add_scalar('Loss/Test', avg_loss, round)
            writer.add_scalar('Loss/Train', train_loss, round)
    writer.flush()

    os.makedirs(args.save_path, exist_ok=True)
    save_DIR = os.path.join(args.save_path, experiment)
    os.makedirs(save_DIR, exist_ok=True)

    log_bin(active_client_bin, partition, save_DIR)
    if args.model_save:
        save_model(model, save_DIR) 
    
    print("=== Centralized Setting ===")
    acc = 0
    center_model = CNN().to(device)
    optimizer = torch.optim.SGD(center_model.parameters(), lr=args.lr, momentum=args.momentum)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in tqdm(range(100)):
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = center_model(x)
            loss = lossf(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()

    acc, central_loss, _ = test_epoch(center_model, test_loader, device)
    print(acc, central_loss)
    

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