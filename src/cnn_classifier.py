import enum
import json
import os
from selectors import EpollSelector
import numpy as np

import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from tqdm import tqdm, trange


from loader import ClassifierTrainTest, TempoSet, ClassifierSet, ClassifierTrainTest

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
        
        elif layer_info == "M":
            all_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        
        elif layer_info == "A":
            all_layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        
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

        if init_weight:
            self._initialize_weights()

        self.num_class = num_class
        return 
    
    def forward(self, x): 
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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

def train(pianoroll_classifier, train_data, device, model_save_dir, num_epoch=30):
    hist_model_loss = []
    hist_model_acc = []

    # declare optimizer
    optimizer = optim.SGD(pianoroll_classifier.parameters(), lr=1e-2, momentum=0.9)

    # declare loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # train
    print("Model training start!")

    train_len = len(train_data)

    for epoch in range(num_epoch):
        print("Current Epoch {}/{}".format(epoch+1, num_epoch))
        print("="*10)

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
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            # record loss and correct prediction number
            running_loss += loss.item() * inputs.size(0) 
            running_correct += torch.sum(output.argmax(dim=1) == labels.data.argmax(dim=1))
        
        # calculate loss and correct prediction number
        epoch_loss = running_loss / len(train_data.dataset)
        epoch_acc = running_correct.double() / len(train_data.dataset)
        
        print("Epoch Loss: {:.4f}, Epoch Accs: {:.4f}".format(epoch_loss, epoch_acc))
        hist_model_loss.append(epoch_loss)
        hist_model_acc.append(epoch_acc)

    print("Training finished!")
    print("="*10)
    print("Training Summary:")
    print("Final Loss: {:.4f}, \nFinal Acc: {:.4f}".format(hist_model_loss[-1], hist_model_acc[-1]))
    print("Saving model...")
    torch.save(pianoroll_classifier.state_dict(), os.path.join(model_save_dir, "pianoroll_classifier.pth"))
    print("Model saved!")

    return 


def test(pianoroll_classifier, test_data, device):
    # declare loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # loss and correct prediction number record
    running_loss = 0.0
    running_correct = 0.0

    # iterate over data

    test_len = len(test_data)

    for inputs, labels in tqdm(test_data, total=test_len):
        # send data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        output = pianoroll_classifier(inputs)
        loss = criterion(output, labels)

        # record loss and correct prediction number
        running_loss += loss.item() * inputs.size(0) 
        running_correct += torch.sum(output.argmax(dim=1) == labels.data.argmax(dim=1))
    
    # calculate loss and correct prediction number
    test_loss = running_loss / len(test_data.dataset)
    test_acc = running_correct.double() / len(test_data.dataset)

    print("Test Loss: {:.4f}, Test Acc: {:.4f}".format(test_loss, test_acc))
    return test_loss, test_acc

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
    train_loader = DataLoader(classifier_train_set, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(classifier_test_set, batch_size=16, shuffle=False, num_workers=4)
    print("Train/Test loader constructed!")
    print("train size: {}, test size: {}\n".format(len(train_loader.dataset), len(test_loader.dataset)))

    # construct model
    feature_cfg = [64, "M", 128, "M", 128, 128, "M"]
    feature_conv_size = 3
    feature_extractor = feature_forger(feature_cfg, need_batch=True, conv_size=feature_conv_size)

    # construct classifier
    classifier_cfg = [512, 256, 128, 64, 32]
    classifier_classifier = classifier_forger(classifier_cfg, num_class=8)
    
    model = PianorollGenreClassifierCNN(feature_extractor, classifier_classifier, num_class=8)
    model.to(device)

    # inspect model
    print("model constructed!\n======== \nmodel structure:\n")
    print(model)
    print("== model summary ==")
    summary(model, (1, 512, 512))
    print("========\n")

    # train model
    model_save_dir = "../model/pianoroll_classifier_cnn"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train(model, train_loader, device, model_save_dir, num_epoch=25)
    print("\ntraining finished!")

    # test model
    test_loss, test_acc = test(model, test_loader, device)
    print("\ntesting finished!")








    





    
    