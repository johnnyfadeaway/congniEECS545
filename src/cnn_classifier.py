import json
from selectors import EpollSelector
import numpy as np

import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader



from loader import ClassifierTrainTest, TempoSet, ClassifierSet, ClassifierTrainTest

def conv_layer(in_dim, out_dim, need_batch=True):
    layer = list()
    layer.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
    if need_batch:
        layer.append(nn.BatchNorm2d(out_dim))
    layer.append(nn.ReLU(inplace=True))
    return layer

def feature_forger(cfg, need_batch=True):
    all_layers = list()
    initial_dim = 1
    for layer_info in cfg:
        if type(layer_info) == int:
            all_layers.extend(conv_layer(initial_dim, layer_info, need_batch))
            previous_dim = layer_info
        elif layer_info == "M":
            all_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        else:
            raise ValueError("Unknown layer type: {}".format(layer_info))
    feature = nn.Sequential(*all_layers)
    return feature




class PianorollGenreClassifierCNN(nn.Module):
    def __init__(self, feature, num_class=8, init_weight=None):
        return 
    
    def forward(self, x):
        pass 
        return 



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

    classifier_set = ClassifierSet(all_data)

    # separate training and testing data
    classifier_idxes = np.array(range(len(classifier_set)))
    np.random.shuffle(classifier_idxes)
    train_idxes = classifier_idxes[:int(len(classifier_idxes) * train_test_split)]
    test_idxes = classifier_idxes[int(len(classifier_idxes) * train_test_split):]

    # construct training and testing data
    classifier_train = ClassifierTrainTest(classifier_set, train_idxes)
    classifier_test = ClassifierTrainTest(classifier_set, test_idxes)

    # construct dataloader
    train_loader = DataLoader(classifier_train, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(classifier_test, batch_size=16, shuffle=False, num_workers=4)

    # construct model





    
    