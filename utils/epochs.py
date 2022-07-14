import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

def get_optimizer(model, args):
    optimizer = None

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    assert optimizer is not None
    return optimizer


def test_epoch(model, dataloader, device='cuda:0', use_pbar=False):
    if use_pbar:
        pbar = tqdm(dataloader, desc='Test epoch')
    else:
        pbar = dataloader

    total, correct = 0, 0
    loss = 0.0

    with torch.no_grad():
        for (imgs, labels) in pbar:
            imgs, labels = imgs.to(device), labels.to(device) 

            outputs = model(imgs)
            _, pred= torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total

    return acc


def train_epoch(model, dataloader, args, device='cuda:0', use_pbar=False):
    if use_pbar:
        pbar = tqdm(dataloader, desc='Train epoch')
    else:
        pbar = dataloader

    total, correct = 0, 0
    total_loss = 0.0

    optimizer = get_optimizer(model, args)
    lossf = nn.CrossEntropyLoss()

    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = lossf(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss