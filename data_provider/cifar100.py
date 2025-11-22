

import os
from PIL import ImageDraw
import random

import numpy as np
from numpy.lib.type_check import _imag_dispatcher
import pandas as pd
import torch
from PIL import Image
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.modules import transformer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json

def load_json_data(file: str) -> list[dict] | dict:
    with open(file=file, mode='r') as f:
        data = json.load(fp=f)
    return data

class ciFAIR100(Dataset):
    def __init__(self, transform, mode, args) -> None:
        root_dir = '../data/cifar100/ciFAIR-100'
        self.transform = transform
        self.root_dir = root_dir
        self.mode = mode
        self.args = args
        test_ground_truth_filename = '../data/cifar100/ciFAIR-100/test.json'
        train_ground_truth_filename = '../data/cifar100/ciFAIR-100/train.json'
        train_noisy_label_filename = '../data/cifar100/ciFAIR-100/synthetic_expert/train_mv.json'
        self.nb_classes = 100
        test_annotator_names = [
                'test_0_1_2_3_4_5_6',
                'test_7_8_9_10_11_12_13',
                'test_14_15_16_17_18_19',
            ]
        train_annotator_names = [
                'train_0_1_2_3_4_5_6',
                'train_7_8_9_10_11_12_13',
                'train_14_15_16_17_18_19',
        ]
        if mode == "test":
            img_path = []
            gt_label = []
            data = load_json_data(test_ground_truth_filename)
            for sample in data:
                img_path.append(sample['file'])
                gt_label.append(sample['label'])
            self.data, self.label = np.array(img_path), np.array(gt_label)
            noise_labels = []
            for annontator_name in test_annotator_names:
                data = load_json_data(file=os.path.join('../data/cifar100/ciFAIR-100/synthetic_expert', annontator_name+'.json'))
                label = []
                for sample in data:
                    label.append(sample['label'])
                noise_labels.append(label)
            self.test_humans = np.array(noise_labels).transpose()
            self.humans = self.test_humans

        elif mode == "train":
            img_path = []
            gt_label = []
            data = load_json_data(train_ground_truth_filename)
            for sample in data:
                img_path.append(sample['file'])
                gt_label.append(sample['label'])
            self.data, self.label = np.array(img_path), np.array(gt_label)
            noise_labels = []
            for annontator_name in train_annotator_names:
                data = load_json_data(file=os.path.join('../data/cifar100/ciFAIR-100/synthetic_expert', annontator_name+'.json'))
                label = []
                for sample in data:
                    label.append(sample['label'])
                noise_labels.append(label)
            self.train_humans = np.array(noise_labels).transpose()

            self.humans = self.train_humans
            
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, self.data[index]))
        img = self.transform(img)
        target = self.label[index]
        humans = self.humans[index]
        return img, target, humans

    def __len__(self):
        return self.data.shape[0]

class cifair100_dataloader():
    def __init__(self, 
                 batch_size=256,
                 num_workers=8,
                 args=None
                 ):
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = ciFAIR100(transform=self.transform_train, 
                                         mode='train', args=self.args)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False)
            return train_loader
    
        elif mode == 'test':
            test_dataset = ciFAIR100(transform=self.transform_test, 
                                          mode='test', args=self.args)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False)
            return test_loader