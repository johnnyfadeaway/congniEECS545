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

def train(generator, gan_loader, logger, device, num_epoch=100):
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
        for x, y  in tqdm(gan_loader):
            x = x.type(torch.FloatTensor)
            x = x[:, 0, :, :]
            x = x.unsqueeze(1)

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
    torch.save(generator.state_dict(), "../model/benchmark_{}.pth".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    
    print("saving history...")
    np.save("../history/hist_benckmark_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_G_loss)
    
    print("saving model finished!")
    return 


class TempoBenchmark(nn.Module):
    def __init__(self, num_in_channels, chunk_len):
        super(TempoBenchmark, self).__init__()

        self.conv1 = nn.Conv2d(num_in_channels, 16, kernel_size=4, stride=4, padding=0)
        self.conv_relu1 = nn.ReLU()
        
        self.deconv1 = nn.ConvTranspose2d(16, 1, kernel_size=(2, 1), stride=(2, 1), dilation=1, padding=0)
        self.deconv_bn = nn.BatchNorm2d(1)
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), dilation=1, padding=0)
        self.deconv_relu = nn.ReLU()

        return 
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_relu1(x)

        x = self.deconv1(x)
        x = self.deconv_bn(x)
        x = self.deconv2(x)
        x = self.deconv_relu(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
    
    gc.collect()
    torch.cuda.empty_cache()

    data_dir = "../data/lpd_5/lpd_5_cleansed"
    all_data = TempoSet()
    all_data.load(data_dir)

    classifier_set = ClassifierSet(all_data, chunk_size=(128*4))
    gan_set = GANdataset(classifier_set)

    gan_loader = DataLoader(gan_set, batch_size=16, shuffle=True, num_workers=2)
    
    baseline = TempoBenchmark(num_in_channels=1, chunk_len=512)
    baseline.to(device)
    print("Baseline model TempoBenchmark constructed!")
    summary(baseline, (1, 512, 512))

    print("attempt training...")
    logger_dir = "../log/benchmark/"
    logger = Logger(logger_dir)
    train(baseline, gan_loader, logger, device, num_epoch=10)

    print("training finished!")