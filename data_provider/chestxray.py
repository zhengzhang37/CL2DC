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


class ChestDataset(Dataset):
    def __init__(self, transform, mode) -> None:
        root_dir = '../data/CXR8'
        self.transform = transform
        self.root_dir = root_dir
        self.mode = mode

        if mode == "test":
            img_list = os.path.join(root_dir, "test_labels.csv")
            gt = pd.read_csv(img_list, index_col=0)
            self.imgs = gt.index.to_numpy()
            gt = np.load(os.path.join('../data/chestxray/test.npy'))[:,-1]
            humans = np.load(os.path.join('../data/chestxray/test.npy'))[:,0:3]
            self.gt = gt.astype(np.int64)
            self.test_humans = humans.astype(np.int64)

            # self.humans = np.array(
            # [
            #     np.random.choice(self.test_humans[i])
            #     for i in range(len(self.test_humans))
            # ])

            self.humans =  self.test_humans

        elif mode == "train":
            img_list = os.path.join(root_dir, "validation_labels.csv")
            gt = pd.read_csv(img_list, index_col=0)
            self.imgs = gt.index.to_numpy()
            gt = np.load(os.path.join('../data/chestxray/val.npy'))[:,-1]
            humans = np.load(os.path.join('../data/chestxray/val.npy'))[:,0:3]
            self.gt = gt.astype(np.int64)
            self.train_humans = humans.astype(np.int64)

            # self.humans = np.array(
            # [
            #     np.random.choice(self.train_humans[i])
            #     for i in range(len(self.train_humans))
            # ])

            self.humans =  self.train_humans
            
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "images", self.imgs[index])
        gt = self.gt[index]
        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_t = self.transform(img)
        humans = self.humans[index]
        return img_t, gt, humans

    def __len__(self):
        return self.imgs.shape[0]

class chestxray_dataloader():
    def __init__(self, 
                 batch_size=256,
                 num_workers=8,
                 args=None
                 ):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (args.dataset.resize[0], args.dataset.resize[0]), scale=(0.2, 1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize(args.dataset.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            )
    def run(self, mode):
        if mode == 'train':
            train_dataset = ChestDataset(transform=self.transform_train, 
                                         mode='train')
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False)
            return train_loader
    
        elif mode == 'test':
            test_dataset = ChestDataset(transform=self.transform_test, 
                                          mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False)
            return test_loader

