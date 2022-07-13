import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model.cnn import CNN
from utils.parser import cifar10_dict, argparser
from utils.datasets import CustomDatasets

def main():
    args = argparser()

    device = 'cuda:0'
    torch.manual_seed(17)

    data_PATH = os.path.join(args.data_dir, 'cifar-10-batches-py')
    train_data, train_labels, test_data, test_labels = cifar10_dict(data_PATH)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32,3)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CustomDatasets(train_data, train_labels, transforms=transform)
    test_dataset = CustomDatasets(test_data, test_labels, transforms=transform)

    # hyperparam
    lr = args.lr
    momentum = args.momentum
    batch_sz = 32
    
    model = CNN().to(device)
    lossf = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=2)
    pbar = tqdm(test_loader, desc='test model')

    total, correct = 0

    with torch.no_grad():
        for (imgs, labels) in pbar:
            imgs, labels = imgs.to(device), labels.to(device) 

            outputs = model(imgs)
            _, pred= torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print((100 * correct / total))

if __name__=='__main__':
    main()