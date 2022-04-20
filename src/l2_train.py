import enum
import json
import os
from selectors import EpollSelector
from matplotlib.font_manager import findfont
from matplotlib import pyplot as plt
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
from generator import generator
from utils import Logger


def train_l2(generator, gan_loader, logger, device, num_epoch=100):
    hist_G_loss = []
    
    # optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
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

        g_running_loss = []

        L1_loss = nn.L1Loss().to(device)
        for x, y in tqdm(gan_loader):
            x = x.type(torch.FloatTensor)
            x, y = x.to(device), y.to(device)

            
            
            # train the generator
            generator.zero_grad()

            generated_result = generator(x)

            # calculate the loss
            g_train_loss = L1_loss(generated_result, y)         
            
            # record loss
            hist_G_loss.append(g_train_loss.detach().item())
            
            # optimizing the model
            g_train_loss.backward()
            g_optimizer.step()
            loss_generator = g_train_loss.detach().item()
            
            g_running_loss.append(loss_generator)
            
        epoch_end_time = time.time()

        # print loss
        print("generator loss {} at epoch {}".format(np.mean(g_running_loss), epoch))

        # write log
        logger.write("generator loss {} at epoch {}".format(np.mean(g_running_loss), epoch))

        logger.write("\nepoch end time: {}\n".format(datetime.now()))
        epoch_used_time = epoch_end_time - epoch_start_time
        epoch_used_hrs = int(epoch_used_time / 3600)
        epoch_used_min = int((epoch_used_time % 3600) / 60)
        epoch_used_sec = int(epoch_used_time % 60)
        logger.write("epoch used time {} hrs {} min {} sec\n".format(epoch_used_hrs, epoch_used_min, epoch_used_sec))
        
    print("training finished!")
    print("saving model...")
    torch.save(generator.state_dict(), "../model/generator_l2_{}.pth".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    
    print("saving history...")
    np.save("../history/hist_G_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_G_loss)
    
    print("saving model finished!")
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


    data_dir = "../data/lpd_5/lpd_5_cleansed"
    all_data = TempoSet()
    all_data.load(data_dir)

    classifier_set = ClassifierSet(all_data, chunk_size=(128*4))
    gan_set = GANdataset(classifier_set)

    gan_loader = DataLoader(gan_set, batch_size=64, shuffle=True, num_workers=1)

    print("initializing the generator model...")
    generator = generator()
    generator.to(device)

    print("initializing the weights...")
    generator.weight_init(mean=0.5, std=0.02)

    print("attept training...")
    print("initalizing logger...")
    logger = Logger("../log/l2_train/".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    train_l2(generator, gan_loader, logger, device, num_epoch=10)

    print("training finished!")