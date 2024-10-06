import torch.utils.data as data
from PIL import Image
import os
import json
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import copy
import random
from torch.utils.data import Dataset, DataLoader

class CHAOYANG(data.Dataset):
    def __init__(self, dataset='chaoyang', root='../data/chaoyang', transform=None, mode='train'):
        self.dataset = dataset
        self.transform = transform
        self.mode = mode
        if dataset == 'chaoyang':
            self.nb_classes = 4
        if self.mode == 'human_test' or self.mode == 'test':
            imgs = []
            labels = []
            labels1 = []
            labels2 = []
            labels3 = []
            json_path = os.path.join('../data/chaoyang/setting/test.json')
            with open(json_path,'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(root,load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
                    labels1.append(load_list[i]["label_A"])
                    labels2.append(load_list[i]["label_B"])
                    labels3.append(load_list[i]["label_C"])
            test_humans = np.concatenate((np.array(labels1)[..., np.newaxis], np.array(labels2)[..., np.newaxis], np.array(labels3)[..., np.newaxis]), 1)
            self.test_data, self.test_labels, self.test_humans = imgs, labels, test_humans
        
        elif self.mode == 'human':
            imgs = []
            labels = []
            labels1 = []
            labels2 = []
            labels3 = []
            json_path = os.path.join('../data/chaoyang/setting/train.json')
            with open(json_path,'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(root,load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
                    labels1.append(load_list[i]["label_A"])
                    labels2.append(load_list[i]["label_B"])
                    labels3.append(load_list[i]["label_C"])
            self.train_human_labels = np.concatenate((np.array(labels1)[..., np.newaxis], np.array(labels2)[..., np.newaxis], np.array(labels3)[..., np.newaxis]), 1)
            self.train_data, self.train_labels = np.array(imgs), np.array(labels)
            self.train_human_predictions = self.train_human_labels


    def __getitem__(self, index):
        if self.mode == 'human':
            img, target = self.train_data[index], self.train_human_predictions[index]
            gt = self.train_labels[index]
            img = Image.open(img)
            img = self.transform(img)
            return img, gt, index
        if self.mode == 'human_test':
            img, target, humans = self.test_data[index], self.test_labels[index], self.test_humans[index]
            img = Image.open(img)
            img = self.transform(img)
            return img, target, humans
        elif self.mode == 'train':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.open(img)
            img = self.transform(img)
            
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.open(img)
            img = self.transform(img)
            return img, target, index

    def __len__(self):
        if self.mode == 'human_test' or self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)



class chaoyang_dataloader():
    def __init__(self, 
                 batch_size=96,
                 num_workers=8,
                 ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
        self.transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = CHAOYANG(transform=self.transform_train, 
                                             mode="human",
                                             )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False)

            return train_loader

        elif mode == 'test':
            test_dataset = CHAOYANG(transform=self.transform_test, 
                                          mode='human_test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False)
            return test_loader

 

 
