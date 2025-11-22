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

class MiceBone(data.Dataset):
    def __init__(self, dataset='micebone', root='../data/MiceBone', transform=None, mode='train'):
        self.dataset = dataset
        self.transform = transform
        self.mode = mode
        self.root = root
        if dataset == 'micebone':
            self.nb_classes = 3
        annotator_names = (
            'Id_47',
            'Id_290',
            'Id_533',
            'Id_534',
            'Id_580',
            'Id_581',
            'Id_966',
            'Id_745'
        )
        root_for_fold = '../data/MiceBone/annotations'
        ground_truth_filename = '../data/MiceBone/annotations/majority_vote.json'
        
        if self.mode == 'test':
        
            img_path = []
            gt_label = []
            with open(file=f'{ground_truth_filename}', mode='r') as f:
                data = json.load(fp=f)
                for sample in data:
                    if sample['file'].split('/')[0] == 'fold5':
                        img_path.append(sample['file'])
                        gt_label.append(sample['label'])
                    
            labels = []
            for annotator_name in annotator_names:
                annotator = osp.join(root_for_fold, annotator_name+'.json')
                with open(file=annotator, mode='r') as f:
                    data = json.load(fp=f)
                    label = []
                    file_to_label = {item['file']: item['label'] for item in data}
                    for i in range(len(img_path)):
                        label.append(file_to_label[img_path[i]])
                labels.append(label)
            labels = np.array(labels).transpose()
            self.test_humans = labels
            self.test_data, self.test_labels = np.array(img_path), np.array(gt_label)

        elif self.mode == 'train':
        
            img_path = []
            gt_label = []
            with open(file=f'{ground_truth_filename}', mode='r') as f:
                data = json.load(fp=f)
                for sample in data:
                    if sample['file'].split('/')[0] != 'fold5':
                        img_path.append(sample['file'])
                        gt_label.append(sample['label'])
                    
            labels = []
            for annotator_name in annotator_names:
                annotator = osp.join(root_for_fold, annotator_name+'.json')
                with open(file=annotator, mode='r') as f:
                    data = json.load(fp=f)
                    label = []
                    file_to_label = {item['file']: item['label'] for item in data}
                    for i in range(len(img_path)):
                        label.append(file_to_label[img_path[i]])
                labels.append(label)
            labels = np.array(labels).transpose()
            self.train_humans = labels
            self.train_data, self.train_labels = np.array(img_path), np.array(gt_label)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.open(osp.join(self.root, img))
            humans = self.train_humans[index]
            img = self.transform(img)
            return img, target, humans
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.open(osp.join(self.root, img))
            humans = self.test_humans[index]
            img = self.transform(img)
            return img, target, humans
    
    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)
        
class micebone_dataloader():
    def __init__(self, 
                 batch_size=256,
                 num_workers=8,
                 ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = MiceBone(transform=self.transform_train, 
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
            test_dataset = MiceBone(transform=self.transform_test, 
                                          mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False)
            return test_loader
        
if __name__ == '__main__':
    micebone = micebone_dataloader()
    train_loader = micebone.run('train')
    test_loader = micebone.run('test')
    for batch_idx, (img, target) in enumerate(test_loader):
        # print(img.shape, target.shape)
        pass