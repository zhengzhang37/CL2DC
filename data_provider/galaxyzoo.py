import json
from collections import Counter
import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
import json
import pickle
import torchvision.transforms as transforms
import copy
import random
from torch.utils.data import Dataset, DataLoader

class GALAXYZOO(data.Dataset):
    def __init__(self, dataset='galaxyzoo', root='../data/galaxy-zoo', transform=None, mode='train'):
        self.dataset = dataset
        self.transform = transform
        self.mode = mode
        self.root = root
        if dataset == 'galaxyzoo':
            self.nb_classes = 2
        test_ground_truth_filename = '../data/galaxy-zoo/test_experts.json'
        train_ground_truth_filename = '../data/galaxy-zoo/train_experts.json'
        if self.mode == 'test':
            img_path = []
            gt_label = []
            exp1 = []
            exp2 = []
            with open(file=test_ground_truth_filename, mode='r') as f:
                data = json.load(fp=f)
                for sample in data:
                    img_path.append(sample['file'])
                    gt_label.append(sample['label'])
                    exp1.append(sample['expert1'])
                    exp2.append(sample['expert2'])
            self.test_data, self.test_labels = img_path, np.array(gt_label)
            experts = np.stack([exp1, exp2], axis=1)
            self.humans = np.stack([exp1, exp2], axis=1)
        
        elif self.mode == 'train':
            img_path = []
            labels = []
            exp1 = []
            exp2 = []
            with open(file=train_ground_truth_filename, mode='r') as f:
                data = json.load(fp=f)
                for sample in data:
                    img_path.append(sample['file'])
                    labels.append(sample['label'])
                    exp1.append(sample['expert1'])
                    exp2.append(sample['expert2'])
            self.train_data, self.train_labels, exp1, exp2 = img_path, np.array(labels), np.array(exp1), np.array(exp2)
            self.humans = np.stack([exp1, exp2], axis=1)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.open(osp.join(self.root, 'images', str(img) + '.jpg'))
            img = self.transform(img)
            humans = self.humans[index]
            return img, target, humans
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_labels[index]
            image = Image.open(osp.join(self.root, 'images', str(img) + '.jpg'))
            image = self.transform(image)
            humans = self.humans[index]
            return image, target, humans
    
    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)
        
class galaxyzoo_dataloader():
    def __init__(self, 
                 batch_size=256,
                 num_workers=8,
                 ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
                                    transforms.RandomRotation(360),
                                    transforms.CenterCrop([256, 256]),
                                    transforms.Resize([128, 128]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ])
        self.transform_test = transforms.Compose([
                                    transforms.CenterCrop([256, 256]),
                                    transforms.Resize([128, 128]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = GALAXYZOO(transform=self.transform_train, 
                                             mode="train",
                                             )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False)

            return train_loader

        elif mode == 'test':
            test_dataset = GALAXYZOO(transform=self.transform_test, 
                                          mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False)
            return test_loader
        
if __name__ == '__main__':
    ham10000 = ham10000_dataloader()
    train_loader = ham10000.run('train')
    test_loader = ham10000.run('test')
    for batch_idx, (img, target) in enumerate(train_loader):
        print(img.shape, target.shape)