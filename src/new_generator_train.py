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
from generator import GeneratorUnet
from discriminator_classic import DiscriminatorClassic

def train_new_gan_with_classic_discriminator(generator, discriminator, gan_loader, logger, device, num_epoch=100):
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

        num_iter = 0
        for x, y in tqdm(gan_loader):
            x = x.type(torch.FloatTensor)
            x, y = x.to(device), y.to(device)

            pitch_height = x.shape[3]
            num_channels = x.shape[1]

            channelized_x = x.reshape(-1, num_channels*4, 512, int(pitch_height/4),)
            # print("DEBUG shape of channelized_x: {}".format(channelized_x.shape))
            # htracks * 4, genre_enlarged * 4, htracks_pos_enc * 4
            genre_channel = channelized_x[:, 4, :, :].unsqueeze(1)
            pos_enc_channel = channelized_x[:, 8, :, :].unsqueeze(1)
            htracks = channelized_x[:, 0:4, :, :]

            x = torch.cat((htracks, genre_channel, pos_enc_channel), dim=1)


            d_optimizer.zero_grad()

            cat_real_d_input = torch.cat([y, x], dim=1)
            # print("DEBUG shape of cat_real_d_input: {}".format(cat_real_d_input.shape))

            d_result_real = discriminator(cat_real_d_input)

            g_result_temp = generator(x)
            cat_fake_d_input = torch.cat([g_result_temp, x], dim=1)
            d_result_fake = discriminator(cat_fake_d_input)

            d_loss_real = BCE_loss(d_result_real, torch.ones_like(d_result_real).to(device))
            d_loss_fake = BCE_loss(d_result_fake, torch.zeros_like(d_result_fake).to(device))

            d_train_loss = d_loss_real * 0.5 + d_loss_fake * 0.5
            d_train_loss.backward()
            d_optimizer.step()
            loss_discriminator = d_train_loss.detach().item()

            # train the generator
            generator.zero_grad()

            generated_result = generator(x)
            d_input = torch.cat([generated_result, x], dim=1)
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
    torch.save(generator.state_dict(), "../model/generator_unet_{}.pth".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    torch.save(discriminator.state_dict(), "../model/discriminator_classic_{}.pth".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    print("saving history...")
    np.save("../history/hist_gen_unet_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_G_loss)
    np.save("../history/hist_gen_unet_l1_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_G_l1_loss)
    np.save("../history/hist_D_loss_classic_{}.npy".format(datetime.now().strftime("%Y%m%d_%H%M%S")), hist_D_loss)
    
    print("saving model finished!")
    return 

if __name__ == "__main__":
    
    print("Program traning U-Net generator with Goodfellow Discriminator start!\n")

    print("=== device report ===")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        current_device = torch.cuda.current_device()
        print("CUDA engaged, Using GPU")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using device: {device_name}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, Using CPU")
    print("=======")

    print("loading dataset...")
    data_dir = "../data/lpd_5/lpd_5_cleansed"
    all_data = TempoSet()
    all_data.load(data_dir)

    classifier_set = ClassifierSet(all_data, chunk_size=(128*4))
    gan_set = GANdataset(classifier_set)
    gan_loader = DataLoader(    gan_set, 
                                batch_size=16, 
                                shuffle=True, 
                                num_workers=0,
                            )
    print("loading dataset finished!")

    print("configuring models...")

    # discriminator config
    discriminator_config = [ 
        ## in_channels(int), out_channels(int), kernel_size(tuple), stride(tuple), padding(tuple), batch_norm(bool)
        (7, 32, (4, 4), (2, 2), (1, 1), False), ## conv1, out size = 256 x 64
        (32, 64, (4, 4), (2, 2), (1, 1), True), ## conv2, out size = 128 x 32
        (64, 128, (4, 4), (2, 2), (1, 1), True), ## conv3, out size = 64 x 16
        (128, 64, (4, 2), (2, 1), (1, 0), True), ## conv4, out size = 32 x 15
        (64, 32, (2, 2), (1, 1), (0, 0), True), ## conv5, out size = 31 x 14
        (32, 16, (2, 1), (1, 1), (0, 0), False), ## conv6, out size = 30 x 14
    ]

    discriminator_classic = DiscriminatorClassic(discriminator_config)
    discriminator_classic.to(device)

    # generator config
    unet_config = {
            "conv": {
                  "conv0": (6, 16, (4,4), (2,2), (1,1)),
                  "conv1": (16, 32, (4,4), (2,2), (1,1)),
                  "conv2": (32, 64, (4,4), (2,2), (1,1)),
                  "conv3": (64, 128, (4,4), (2,2), (1,1)),
                  "conv4": (128, 256, (4,4), (2,2), (1,1)),
                  "conv5": (256, 256, (4,4), (2,2), (1,1)),
            },
            "deconv": {
                  "deconv0": (256, 256, (4,4), (2,2), (1,1), (1,1)),
                  "deconv1": (256*2, 128, (4,4), (2,2), (1,1), (1,1)),
                  "deconv2": (128*2, 64, (4,4), (2,2), (1,1), (1,1)),
                  "deconv3": (64*2, 32, (4,4), (2,2), (1,1), (1,1)),
                  "deconv4": (32*2, 16, (4,4), (2,2), (1,1), (1,1)),
                  "deconv5": (16*2, 1, (4,4), (2,2), (1,1), (1,1)),
            }
    }

    seq = {
            "conv": ["conv0", "conv1", "conv2", "conv3", "conv4", "conv5"],
            "deconv": ["deconv0", "deconv1", "deconv2", "deconv3", "deconv4", "deconv5"]
    }

    generator_unet = GeneratorUnet(unet_config, seq)
    generator_unet.to(device)

    print("configuring models finished!")
    print("=== model report ===")
    print("generator_unet:")
    summary(generator_unet, (6, 512, 128))
    print("\ndiscriminator_classic:")
    summary(discriminator_classic, (7, 512, 128))
    print("=======")

    # attempt training
    logger = Logger('../log/unet/')
    print("training start!")
    train_new_gan_with_classic_discriminator(generator_unet, discriminator_classic, gan_loader, logger, device, num_epoch=20)

    print("training finished!")
    print("Program finished!")
    print("=======")
    

