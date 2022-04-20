import os
import numpy as np
from datetime import datetime
import time
import math
import gc
from pypianoroll import pitch_range

import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from tqdm import tqdm, trange

from loader import ClassifierTrainTest, TempoSet, ClassifierSet, ClassifierTrainTest, GANdataset
from utils import Logger, normal_init
from generator import generator


def discriminator_layer_forger(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), batch_norm=True):
    layer = nn.ModuleList()
    layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    if batch_norm:
        layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.LeakyReLU(0.2, inplace=True))
    return layer 

def train_gan_with_classic_discriminator(generator, discriminator, gan_dataset, logger, device, num_epoch=100):
    hist_G_loss = []
    hist_D_loss = []
    hist_G_l1_loss = []
    
    # optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print("training start!")

    # write log
    logger.write("\n\n==========\n")
    logger.write("training started on {}\n".format(datetime.now()))
    logger.write("total epoch: {}\n".format(num_epoch))
    logger.write("==========\n\n")

    training_start_time = time.time()

    for epoch in range(num_epoch):
        print("epoch: {}/{}".format(epoch, num_epoch))
        
        # write log
        logger.write("\n=======")
        logger.write("epoch: {}/{}\n".format(epoch, num_epoch))
        logger.write("=======\n")
        logger.write("epoch start time: {}\n".format(datetime.now()))
        logger.write("currently used memory on GPU: {}\n".format(torch.cuda.memory_allocated()))
        
        # calculate currently used time to hrs min and sec                              
        used_time = time.time() - training_start_time
        used_hrs = int(used_time / 3600)
        used_min = int((used_time % 3600) / 60)
        used_sec = int(used_time % 60)
        logger.write("used time: {}hrs {}min {}sec\n".format(used_hrs, used_min, used_sec))

        epoch_start_time = time.time()

        d_running_loss = []
        g_running_loss = []

        BCE_loss = nn.BCELoss().to(device)
        L1_loss = nn.L1Loss().to(device)

        gan_dataset_len = len(gan_dataset)
        
        for i in tqdm(range(gan_dataset_len), total=gan_dataset_len):

            x, y = gan_dataset[i]
            
            x = x.type(torch.FloatTensor)
            x, y = x.to(device), y.to(device)

            pitch_height = x.shape[3]
            num_channels = x.shape[1]

            channelized_x = x.reshape(-1, num_channels*4, 512, int(pitch_height/4),)

            d_optimizer.zero_grad()

            cat_real_d_input = torch.cat([y, channelized_x], dim=1)

            d_result_real = discriminator(cat_real_d_input)

            g_result_temp = generator(x)
            cat_fake_d_input = torch.cat([g_result_temp, channelized_x], dim=1)
            d_result_fake = discriminator(cat_fake_d_input)

            d_loss_real = BCE_loss(d_result_real, torch.ones_like(d_result_real).to(device))
            d_loss_fake = BCE_loss(d_result_fake, torch.zeros_like(d_result_fake).to(device))

            d_train_loss = d_loss_real * 0.5 + d_loss_fake * 0.5
            d_train_loss.backward()
            d_optimizer.step()
            loss_discriminator = d_train_loss.detach().item()

            # train the generator
            g_optimizer.zero_grad()

            generated_result = generator(x)
            d_input = torch.cat([generated_result, channelized_x], dim=1)
            d_result = discriminator(d_input)

            g_train_loss = BCE_loss(d_result, torch.ones_like(d_result).to(device)) + 100 * L1_loss(generated_result, y)

            # record loss
            hist_G_loss.append(g_train_loss.detach().item())
            hist_G_l1_loss.append(L1_loss(generated_result, y).detach().item())
            hist_D_loss.append(loss_discriminator)

            # optimizing the model
            g_train_loss.backward()
            g_optimizer.step()
            loss_generator = g_train_loss.detach().item()
            
            d_running_loss.append(loss_discriminator)
            g_running_loss.append(loss_generator)
            
        epoch_end_time = time.time()

        # print loss
        print("discriminator loss {} at epoch {}".format(np.mean(d_running_loss), epoch))
        print("generator loss {} at epoch {}".format(np.mean(g_running_loss), epoch))

        # write log
        logger.write("discriminator loss {} at epoch {}".format(np.mean(d_running_loss), epoch))
        logger.write("generator loss {} at epoch {}".format(np.mean(g_running_loss), epoch))

        logger.write("\nepoch end time: {}\n".format(datetime.now()))
        epoch_used_time = epoch_end_time - epoch_start_time
        epoch_used_hrs = int(epoch_used_time / 3600)
        epoch_used_min = int((epoch_used_time % 3600) / 60)
        epoch_used_sec = int(epoch_used_time % 60)
        logger.write("epoch used time {} hrs {} min {} sec\n".format(epoch_used_hrs, epoch_used_min, epoch_used_sec))
        
    print("training finished!")
    print("saving model...")
    torch.save(generator.state_dict(), "../model/generator_classic_{}.pth".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    torch.save(discriminator.state_dict(), "../model/discriminator_classic_{}.pth".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    print("saving history...")
    np.save("../history/hist_G_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_G_loss)
    np.save("../history/hist_G_l1_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_G_l1_loss)
    np.save("../history/hist_D_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_D_loss)
    
    print("saving model finished!")
    return 

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
        return x
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
        return 

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
        (13, 32, (4, 4), (2, 2), (1, 1), False), ## conv1, out size = 256 x 64
        (32, 64, (4, 4), (2, 2), (1, 1), True), ## conv2, out size = 128 x 32
        (64, 128, (4, 4), (2, 2), (1, 1), True), ## conv3, out size = 64 x 16
        (128, 256, (4, 2), (2, 1), (1, 0), True), ## conv4, out size = 32 x 15
        (256, 128, (2, 2), (1, 1), (0, 0), True), ## conv5, out size = 31 x 14
        (128, 64, (2, 1), (1, 1), (0, 0), False), ## conv6, out size = 30 x 14
        
    ]

    data_dir = "../data/lpd_5/lpd_5_cleansed"
    all_data = TempoSet()
    all_data.load(data_dir)

    classifier_set = ClassifierSet(all_data, chunk_size=(128*4))
    gan_set = GANdataset(classifier_set)
    
    discriminator_classic = DiscriminatorClassic(discriminator_config)
    discriminator_classic.to(device)
    print("discriminator constructed!")
    print("inspecting the model...")
    print(discriminator_classic)
    print("summary of the model...")
    summary(discriminator_classic, (13, 512, 128))

    print("initializing the generator model...")
    generator = generator()
    generator.to(device)

    print("initializing the weights...")
    discriminator_classic.weight_init(mean=0.5, std=0.02)
    generator.weight_init(mean=0.5, std=0.02)

    print("attept training...")
    print("initalizing logger...")
    logger = Logger("../log/discriminator_classic/train_log_classic_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    train_gan_with_classic_discriminator(generator, discriminator_classic, gan_set, logger, device=device, num_epoch=10)

    print("training finished!")
    print("exiting...")





