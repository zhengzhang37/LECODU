import os
import torch
import copy
import random
import json
from utils.randaug import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os.path as osp

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset):
    def __init__(self, dataset, noise_type, mode, user_nums, transform, transform_s=None, r=None, idn_dir='../data/cifar-10h'):
        self.dataset = dataset
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.user_nums = user_nums
        self.noise_type = noise_type
        self.r = r
        if dataset == 'cifar':
            self.nb_classes = 10
        if self.mode == 'test':
            self.test_data = np.load(osp.join(idn_dir, 'cifar-10-npy', 'test_images.npy'))
            self.test_label = np.load(osp.join(idn_dir, 'cifar-10-npy', 'test_labels.npy'))
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
            self.test_label_prob = np.load(idn_dir + "../data/cifar10h-probs.npy")
            self.test_label_count = np.load(idn_dir + "../data/cifar10h-counts.npy")
            if self.r == 0.2:
                self.test_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'test_idn20.npy'))[:user_nums])
            elif self.r == 0.3:
                self.test_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'test_idn30.npy'))[:user_nums])
            elif self.r == 0.4:
                self.test_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'test_idn40.npy'))[:user_nums])
            elif self.r == 0.5:
                self.test_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'test_idn50.npy'))[:user_nums])
            elif self.r == 1:
                self.test_humans = np.load(osp.join(idn_dir, 'idn-100-users', 'test_cifarn.npy'))[:,:user_nums]
        else:
            self.train_data = np.load(osp.join(idn_dir, 'cifar-10-npy', 'train_images.npy'))
            self.train_label = np.load(osp.join(idn_dir, 'cifar-10-npy', 'train_labels.npy'))
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            self.prob = np.load(os.path.join('../pseudo_label/probs_1.npy'))
            self.prob2 = np.load(os.path.join('../pseudo_label/probs_2.npy'))
            self.threshold = []
            for i in range(len(self.prob)):
                if self.prob[i] < 0.5 or self.prob2[i] < 0.5:
                    self.threshold.append(i)
            self.threshold = np.array(self.threshold)
            self.train_clean_data = np.delete(self.train_data, self.threshold, 0)
            train_noisy_label1 = np.load('../pseudo_label/pseudo_labels_1.npy')
            train_noisy_label2 = np.load('../pseudo_label/pseudo_labels_2.npy')[..., np.newaxis]
            # self.train_noisy_labels = np.concatenate((train_noisy_label1, train_noisy_label2), 1)
            self.train_noisy_labels = train_noisy_label1
            # self.train_clean_labels = np.delete(self.train_noisy_labels, self.threshold, 0)
            # print("clean train data: ", len(self.train_clean_data))
            if self.r == 0.2:
                self.train_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'train_idn20.npy'))[:user_nums])
            elif self.r == 0.3:
                self.train_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'train_idn30.npy'))[:user_nums])
            elif self.r == 0.4:
                self.train_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'train_idn40.npy'))[:user_nums])
            elif self.r == 0.5:
                self.train_humans = np.transpose(np.load(osp.join(idn_dir, 'idn-100-users', 'train_idn50.npy'))[:user_nums])
            elif self.r == 1:
                self.train_humans = np.load(osp.join(idn_dir, 'idn-100-users', 'train_cifarn.npy'))[:,:user_nums]
            # self.train_clean_humans = np.delete(self.train_humans, self.threshold, 0)

    def __getitem__(self, index):
        if self.mode == 'test':
            img, target, humans = self.test_data[index], self.test_label[index], self.test_humans[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, humans
        
        elif self.mode == 'train':
            img, target, humans = self.train_data[index], self.train_noisy_labels[index], self.train_humans[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)

            # return img1, img2, target, humans, index

            return img1, target, humans

    def __len__(self):
        if self.mode == 'test' or self.mode == 'cifar10h':
            return len(self.test_data)
        else:
            return len(self.train_data)

class cifar_dataloader():
    def __init__(self, 
                 dataset='cifar',
                 noise_type = 'idn',
                 batch_size=256,
                 num_workers=8,
                 user_nums=10,
                 r = 0.2,
                 ):
        self.r = r
        self.user_nums = user_nums
        self.dataset = dataset
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.dataset == 'cifar':
            self.transform_train = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.transform_train_s = copy.deepcopy(self.transform_train)
            self.transform_train_s.transforms.insert(0, RandomAugment(3,5))
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
    
    def run(self, mode):
        if mode == 'train':
            if self.noise_type == 'idn':
                train_dataset = cifar_dataset(dataset=self.dataset, 
                                        noise_type=self.noise_type, 
                                        mode='train', 
                                        user_nums=self.user_nums, 
                                        transform=self.transform_train, 
                                        transform_s=self.transform_train_s,
                                        r=self.r)
                trainloader = DataLoader(
                    dataset=train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True)
            return trainloader

        elif mode == 'test':
            if self.noise_type == 'idn':
                test_dataset = cifar_dataset(dataset=self.dataset, 
                                             noise_type=self.noise_type, 
                                             mode='test', 
                                             user_nums=self.user_nums, 
                                             transform=self.transform_test, 
                                             transform_s=None, 
                                             r=self.r)
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers)
                return test_loader

