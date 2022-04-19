import os
import numpy as np
from datetime import datetime
import time
import math
import gc

import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from tqdm import tqdm, trange

from loader import ClassifierTrainTest, TempoSet, ClassifierSet, ClassifierTrainTest, GANdataset
from utils import Logger

def discriminator_layer_forger(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), batch_norm=True):
    layer = nn.ModuleList()
    layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    if batch_norm:
        layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.LeakyReLU(0.2, inplace=True))
    return layer 

class DiscriminatorClassic(nn.Module):
    def __init__(self, discriminator_config, ):
        super(DiscriminatorClassic, self).__init__()

        self.layers = nn.ModuleList()

        for layer_info in discriminator_config:
            in_channels, out_channels, kernel_size, stride, padding, batch_norm = layer_info
            self.layers.extend(discriminator_layer_forger(in_channels, out_channels, kernel_size, stride, padding, batch_norm))
        
        # add last layer
        self.layers.append(nn.Conv2d(out_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.layers.append(nn.Sigmoid())

        """
        self.layer = discriminator_layer_forger(1, 32, batch_norm=False)
        ## out size of conv1 = 256 x 64
        self.layer2 = discriminator_layer_forger(32, 64)
        ## out size of conv2 = 128 x 32
        self.layer3 = discriminator_layer_forger(64, 128)
        ## out size of conv3 = 64 x 16
        self.layer4 = discriminator_layer_forger(128, 256)
        ## out size of conv4 = 32 x 8
        self.layer5 = 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(4, 2), stride=(2, 1), padding=(1, 0))
        self.bn5 = nn.BatchNorm2d(512)
        self.active5 = nn.LeakyReLU(0.2)
        # width (8 - 2 + 0) / 1 + 1 = 8
        ## out size of conv5 = 16 x 8
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(4, 2), stride=(2, 2), padding=(1, 0))
        self.bn6 = nn.BatchNorm2d(512)
        self.active6 = nn.LeakyReLU(0.2)
        ## out size of conv6 = 8 x 4
        self.conv7 = nn.Conv2d(512, 256, kernel_size=(4, 2), stride=(2, 2), padding=(1, 0))
        """
        return 
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)


if __name__ == "__main__":

    print("======\ndevice report: ")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        current_device = torch.cuda.current_device()
        print("CUDA engaged, Using GPU")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using device: {device_name}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, Using CPU")
    print("======\n")

    discriminator_config = [ 
        ## in_channels(int), out_channels(int), kernel_size(tuple), stride(tuple), padding(tuple), batch_norm(bool)
        (1, 32, (4, 4), (2, 2), (1, 1), False), ## conv1, out size = 256 x 64
        (32, 64, (4, 4), (2, 2), (1, 1), True), ## conv2, out size = 128 x 32
        (64, 128, (4, 4), (2, 2), (1, 1), True), ## conv3, out size = 64 x 16
        (128, 256, (4, 2), (2, 1), (1, 0), True), ## conv4, out size = 32 x 15
        (256, 128, (2, 2), (1, 1), (0, 0), True), ## conv5, out size = 31 x 14
        (128, 64, (2, 1), (1, 1), (0, 0), False), ## conv6, out size = 30 x 14
        
    ]
    
    discriminator_classic = DiscriminatorClassic(discriminator_config)
    discriminator_classic.to(device)
    print("discriminator constructed!")
    print("inspecting the model...")
    print(discriminator_classic)
    print("summary of the model...")
    summary(discriminator_classic, (1, 512, 128))



