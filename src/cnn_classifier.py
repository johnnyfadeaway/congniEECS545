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


from loader import ClassifierTrainTest, TempoSet, ClassifierSet, ClassifierTrainTest
import cnnclfr_config as config
from utils import Logger

   

def conv_layer(in_dim, out_dim, need_batch=True, conv_kernel_size=4):
    """
    VGG like convolutional layer constructor
    Following conventional:
        conv-batch-relu

    Args:
        in_dim (int, ): input dimension
        out_dim (int, ): output dimension
        need_batch (bool, optional): whether to add batch normalization layer. Defaults to True.
        conv_kernel_size (int, optional): kernel size of convolutional layer. Defaults to 4.
    
    Returns:
        layer (nn.ModuleList, ): list of nn.Module
    """
    layer = nn.ModuleList()
    layer.append(nn.Conv2d(in_dim, out_dim, kernel_size=conv_kernel_size, stride=1, padding=1))
    if need_batch:
        layer.append(nn.BatchNorm2d(out_dim))
    layer.append(nn.ReLU(inplace=True))
    return layer

def feature_forger(cfg, need_batch=True, conv_size=4):
    """
    function for construct feature extraction network
    Also adapted from VGG
    
    Args:
        cfg (list, ): list of layer configuration
        need_batch (bool, optional): whether to add batch normalization layer. Defaults to True.
        conv_size (int, optional): kernel size of convolutional layer. Defaults to 4.
    
    Returns:
        all_layer (nn.Sequential, ): sequential of all layers
    """
    all_layers = nn.ModuleList()
    previous_dim = 1
    
    for i, layer_info in enumerate(cfg):
        
        if type(layer_info) == int:
            if i == 0:
                all_layers.extend(conv_layer(previous_dim, layer_info, need_batch=False, conv_kernel_size=conv_size))
            else:
                all_layers.extend(conv_layer(previous_dim, layer_info, need_batch, conv_kernel_size=conv_size))
            previous_dim = layer_info
        
        elif layer_info[0] == "M":
            kernel_size = int(layer_info[1:]) 
            all_layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=0, ceil_mode=False))
        
        elif layer_info[0] == "A":
            kernel_size = int(layer_info[1:])
            all_layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=0, ceil_mode=False))
        
        else:
            raise ValueError("Unknown layer type: {}".format(layer_info))
    return nn.Sequential(*all_layers)

def linear_layer(in_dim, out_dim):
    """
    Linear layer constructor
    
    Args:
        in_dim (int, ): input dimension
        out_dim (int, ): output dimension
    
    Returns:
        layer (nn.ModuleList, ): list of nn.Module
    """
    layer = nn.ModuleList()
    layer.append(nn.Linear(in_dim, out_dim, bias=True))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Dropout(p=0.3, inplace=False))
    return layer

def classifier_forger(cfg, num_class):
    """
    function for construct classifier network
    Also adapted from VGG
    
    Args:
        cfg (list, ): list of layer configuration
    
    Returns:
        all_layer (nn.Sequential, ): sequential of all layers
    """
    all_layers = nn.ModuleList()
    previous_dim = 3200
    for layer_info in cfg:
        all_layers.extend(linear_layer(previous_dim, layer_info))
        previous_dim = layer_info
    all_layers.append(nn.Linear(previous_dim, num_class))    
    return nn.Sequential(*all_layers)


class PianorollGenreClassifierCNN(nn.Module):
    def __init__(self, feature, classifier, num_class=8, init_weight=None):
        super(PianorollGenreClassifierCNN, self).__init__()
        
        self.feature = feature 
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = classifier
        self.softmax = nn.Softmax(dim=1)

        if init_weight:
            self._initialize_weights()

        self.num_class = num_class
        return 
    
    def forward(self, x): 
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.softmax(x)
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def train(pianoroll_classifier, train_data, device, model_save_dir, recorder, num_epoch=30):
    hist_model_loss = []
    hist_model_acc = []

    # declare optimizer
    optimizer = optim.SGD(pianoroll_classifier.parameters(), lr=1e-2, momentum=0.9)

    # declare loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # train
    print("Model training start!")

    train_len = len(train_data)
    date_hour_info = "{}_{}".format(datetime.now().date(), datetime.now().hour)

    
    recorder.write("training started on {}\n".format(datetime.now()))
    train_start_time = time.time()

    
    for epoch in range(num_epoch):
        
        time_epoch_start = time.time()
        used_time = time_epoch_start - train_start_time
        used_hrs = used_time // 3600
        used_mins = (used_time - 3600 * used_hrs) // 60
        used_secs = used_time - 3600 * used_hrs - 60 * used_mins

        print("Current Epoch {}/{}".format(epoch+1, num_epoch))
        print("Curretly used time: {}h {}m {:.2f}s".format(used_hrs, used_mins, used_secs))
        print("="*10)

        
        recorder.write("="*10 + "\n")
        
        recorder.write("Current Epoch {}/{}\n".format(epoch+1, num_epoch))
        recorder.write("Entry on {}\n".format(time.asctime(time.localtime(time.time()))))
        recorder.write("Currently used time: {}h {}m {:.2f}s\n".format(used_hrs, used_mins, used_secs))
        
        # loss and correct prediction number record
        running_loss = 0.0
        running_correct = 0.0

        # iterate over data
        
        for inputs, labels in tqdm(train_data, total=train_len):
            # send data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = pianoroll_classifier(inputs)
            # print("DEBUG output: {}".format(output.shape))
            # print("DEBUG labels: {}".format(labels.shape))
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            # record loss and correct prediction number
            running_loss += loss.detach().item() # * inputs.size(0) 
            # print("DEBUG output: {}".format(output.argmax(dim=1)))
            # print("DEBUG ground truth: {}".format(labels.data.argmax(dim=1)))
            # print("DEBUG correct: {}".format((output.argmax(dim=1) == labels.data.argmax(dim=1)).int()))
            running_correct += torch.sum(output.argmax(dim=1) == labels.data.argmax(dim=1))
        
        # calculate loss and correct prediction number
        
        epoch_loss = running_loss / len(train_data.dataset)
        epoch_acc = running_correct.double() / len(train_data.dataset)

        time_epoch_end = time.time()
        
        print("Epoch Loss: {:.4f}, Epoch Accs: {:.4f}".format(epoch_loss, epoch_acc))
        recorder.write("Epoch Loss: {:.4f}, Epoch Accs: {:.4f}\n".format(epoch_loss, epoch_acc))
        time_used_epoch = round(time_epoch_end - time_epoch_start) 
        recorder.write("Epoch Time: {}:{}\n".format(time_used_epoch // 60, time_used_epoch % 60))

        hist_model_loss.append(epoch_loss)
        hist_model_acc.append(epoch_acc)

    train_end_time = time.time()
    train_total_time = round(train_end_time - train_start_time)
    train_total_hours = train_total_time // 3600
    train_total_minutes = (train_total_time - train_total_hours * 3600) // 60
    train_total_seconds = train_total_time - train_total_hours * 3600 - train_total_minutes * 60
    sec_per_epoch = round(train_total_time / num_epoch)

    print("Training finished!")
    print("="*10)
    print("Training Summary:")
    print("Final Loss: {:.4f}, \nFinal Acc: {:.4f}".format(hist_model_loss[-1], hist_model_acc[-1]))
    
    recorder.write("\nTraining Finished\nTraining Summary:\n")
    recorder.write("Final Loss: {:.4f}, \nFinal Acc: {:.4f}\n".format(hist_model_loss[-1], hist_model_acc[-1]))
    recorder.write("\ntraining ended on {}\n".format(datetime.now()))
    recorder.write("training total time: {} hours {} minutes {} seconds\n".format(train_total_hours, train_total_minutes, train_total_seconds))
    recorder.write("with average time of {} min {} sec per epoch\n".format(sec_per_epoch//60, sec_per_epoch%60))
    recorder.write("="*10+"\n")

    print("Saving model...")
    model_save_dir = os.path.join(model_save_dir, "pianoroll_classifier_{}.pth".format(date_hour_info))
    torch.save(pianoroll_classifier.state_dict(), model_save_dir)
    print("Model saved!")

    recorder.write("Model saved\n")
    recorder.write("model saved at {}\n".format(model_save_dir))

    return hist_model_loss, hist_model_acc


def test(pianoroll_classifier, test_data, device):
    
    # correct prediction number record
    running_correct = 0.0

    # iterate over data

    test_len = len(test_data)

    for inputs, labels in tqdm(test_data, total=test_len):
        # send data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        output = pianoroll_classifier(inputs)
        
        # record los and correct prediction number
        running_correct += torch.sum(output.argmax(dim=1) == labels.data.argmax(dim=1))
    
    # calculate loss and correct prediction number
    test_acc = running_correct.double() / len(test_data.dataset)

    print("Test Acc: {:.4f}".format(test_acc))
    return test_acc

if __name__ == "__main__":

    print("cnn classifier start!")
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
    train_test_split = 0.7
    
    all_data = TempoSet()
    all_data.load(data_dir)

    classifier_set = ClassifierSet(all_data, chunk_size=(128*4))

    # separate training and testing data
    classifier_idxes = np.array(range(len(classifier_set)))
    np.random.shuffle(classifier_idxes)
    train_idxes = classifier_idxes[:int(len(classifier_idxes) * train_test_split)]
    test_idxes = classifier_idxes[int(len(classifier_idxes) * train_test_split):]

    # construct training and testing data
    classifier_train_set = ClassifierTrainTest(classifier_set, train_idxes)
    classifier_test_set = ClassifierTrainTest(classifier_set, test_idxes)

    # construct dataloader
    train_loader = DataLoader(classifier_train_set, 
                                batch_size=config.batch_size, 
                                shuffle=True, 
                                num_workers=config.num_workers)  
    test_loader = DataLoader(classifier_test_set, 
                                batch_size=16, 
                                shuffle=False, 
                                num_workers=config.num_workers)
    
    
    
    print("Train/Test loader constructed!")
    print("train size: {}, test size: {}\n".format(len(train_loader.dataset), len(test_loader.dataset)))



    # construct model
    feature_cfg = config.feature_cfg
    feature_conv_size = config.feature_conv_size
    feature_extractor = feature_forger(feature_cfg, need_batch=True, conv_size=feature_conv_size)

    # construct classifier
    classifier_cfg = config.classifier_cfg
    classifier_classifier = classifier_forger(classifier_cfg, num_class=8)
    
    model = PianorollGenreClassifierCNN(feature_extractor, classifier_classifier, num_class=8)
    model.to(device)

    # inspect model
    print("model constructed!\n======== \nmodel structure:\n")
    print(model)
    print("== model summary ==")
    model_stat = summary(model, (1, 512, 512))
    print("========\n")

    # train model
    model_save_dir = "../model/pianoroll_classifier_cnn"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    log_dir = "../log/pianoroll_classifier_cnn"
    logger = Logger(log_dir)
    print("logger constructed!")
    print("logger located at {}\n".format(logger.current_log_dir))

    torch.cuda.empty_cache()
    gc.collect()
    print("gpu memory cache emptied!")

    # general config and env info
    # write into logger
    logger.write("\n======\nGeneral Training info:\n")
    logger.write("device: {}\n".format(device))
    logger.write("device name: {}\n".format(device_name))
    logger.write("model save dir: {}\n".format(model_save_dir))
    logger.write("log dir: {}\n".format(log_dir))
    
    logger.write("total num of epochs: {}\n".format(config.num_epoch))
    logger.write("batch size: {}\n".format(config.batch_size))
    logger.write("num of workers: {}\n".format(config.num_workers))
    logger.write("feature conv size: {}\n".format(config.feature_conv_size))
    logger.write("==========\n")

    # log network structure
    logger.write("\n======\nModel info:\n")
    logger.write("model structure:\n")
    logger.write(str(model))
    logger.write("\nmodel summary:\n")
    logger.write(str(model_stat))
    logger.write("==========\n")
    
    print("starting training...")
    try:
        hist_loss, hist_acc = train(model, train_loader, device, model_save_dir, logger, num_epoch=config.num_epoch)
    except Exception as errormsg:
        print(errormsg)
        logger.write("\n********\nERROR ENCOUNTERED\n")
        logger.write("\n"+str(errormsg)+"\n")
        logger.write("training stopped on {}\n".format(datetime.now()))
        logger.write("training failed!\n")
        print("Training failed!")

    print("\ntraining finished!")



    # test model
    test_acc = test(model, test_loader, device)
    print("\ntesting finished!")
    print("test acc: {}".format(test_acc))
    logger.write("\n======\nTest info:\n")
    logger.write("test loss: {}, test acc: {}\n".format(test_loss, test_acc))
    logger.write("testing finished!\n")

    logger.save_hist(hist_loss, hist_acc)

    # plot and save training curve
    x_range = np.arange(0, config.num_epoch)
    fig, ax = plt.subplots()
    ax.plot(x_range, hist_loss, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(log_dir, "train_loss_{}.png".format(date_hour_info)))


    logger.write("=============")

