import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self, num_classes, in_channel=3):
        super().__init__()
        
        # conv1
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # conv2
        self.conv2 = nn.Conv2d(6, 16, 5)

        # FC
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class NaiveCNN(nn.Module):
    # From FedCor
    def __init__(self, args, input_shape = [3,32,32], num_classes=10, final_pool=False):
        super(NaiveCNN, self).__init__()
        self.convs = []
        self.fcs = []
        self.final_pool=final_pool
        if len(args.kernel_sizes) < len(args.num_filters):
            exlist = [args.kernel_sizes[-1] for i in range(len(args.num_filters)-len(args.kernel_sizes))]
            args.kernel_sizes.extend(exlist)
        elif len(args.kernel_sizes) > len(args.num_filters):
            exlist = [args.num_filters[-1] for i in range(len(args.kernel_sizes)-len(args.num_filters))]
            args.num_filters.extend(exlist)
        output_shape = np.array(input_shape)
        for ksize in args.kernel_sizes[:-1] if not final_pool else args.kernel_sizes:
            if args.padding:
                pad = ksize//2
                output_shape[1:] = (output_shape[1:]+2*pad-ksize-1)//2+1
            else:
                output_shape[1:] = (output_shape[1:]-ksize-1)//2+1
        if not final_pool:
            if args.padding:
                pad = args.kernel_sizes[-1]//2
                output_shape[1:] = output_shape[1:]+2*pad-args.kernel_sizes[-1]+1
            else:
                output_shape[1:] = output_shape[1:]-args.kernel_sizes[-1]+1
        output_shape[0] = args.num_filters[-1]
        conv_out_length = output_shape[0]*output_shape[1]*output_shape[2]
        
        self.convs.append(nn.Conv2d(input_shape[0], args.num_filters[0], kernel_size=args.kernel_sizes[0],padding = args.kernel_sizes[0]//2 if args.padding else 0))
        for n in range(len(args.num_filters)-1):
            self.convs.append(nn.Conv2d(args.num_filters[n], args.num_filters[n+1], kernel_size=args.kernel_sizes[n+1],padding = args.kernel_sizes[n+1]//2 if args.padding else 0))
        #self.conv2_drop = nn.Dropout2d()
        self.fcs.append(nn.Linear(conv_out_length, args.mlp_layers[0]))
        for n in range(len(args.mlp_layers)-1):
            self.fcs.append(nn.Linear(args.mlp_layers[n], args.mlp_layers[n+1]))
        self.fcs.append(nn.Linear(args.mlp_layers[-1], num_classes))
        
        self.convs = nn.ModuleList(self.convs)
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for n in range(len(self.convs)-1 if not self.final_pool else len(self.convs)):
            x = F.relu(F.max_pool2d(self.convs[n](x), 2))
        if not self.final_pool:
            x = F.relu(self.convs[-1](x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        for n in range(len(self.fcs)-1):
            x = F.relu(self.fcs[n](x))
            #x = F.dropout(x, training=self.training)
        x = self.fcs[-1](x)
        return x