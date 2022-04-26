from lib2to3.pgen2.tokenize import tokenize
from transformers import PerceiverConfig, PerceiverModel, TrainingArguments
from transformers import Trainer
from transformers import PerceiverFeatureExtractor, PerceiverForImageClassificationLearned
from datasets import metric

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

from tqdm import tqdm, trange

import torch 
from torch import optim as optim
from torch.utils.data import DataLoader

from loader import ClassifierTrainTest, TempoSet, ClassifierSet, ClassifierTrainTest
import perceiver_config as config
from utils import Logger


def compute_metric(eval_pred):
    predictions = np.argmax(eval_pred.preditions, axis=1)
    labels = np.argmax(eval_pred.label_ids, axis=1)
    return metric.compute(predictions, reference=labels)

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
    train_test_split = 0.7
    
    all_data = TempoSet()
    all_data.load(data_dir)

    classifier_set = ClassifierSet(all_data, chunk_size=512)

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
                                batch_size=config.batch_size, 
                                shuffle=False, 
                                num_workers=config.num_workers)

    # construct model
    feature_extractor = PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-learned")
    model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
    
    model.to(device)
    feature_extractor.to(device)

    train_args = TrainingArguments(
        output_dir = "../model/perceiver/",
        evaluation_strategy="epoch",
        do_train=True,
        num_train_epochs=2,
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epoch,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        radient_accumulation_steps=4,
    )

    # construct trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset = classifier_train_set,
        eval_dataset = classifier_test_set,
        tokenizer = feature_extractor,
        logging_steps=10,
    )


