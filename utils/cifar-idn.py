import numpy as np
import torch.utils.data as Data
from PIL import Image
import torch
import tools
import os
num_classes=10
feature_size=3*32*32 
norm_std=0.1
noise_rate=0.2

original_images = np.load('../data/cifar-10h/cifar-10-npy/test_images.npy')
original_labels = np.load('../data/cifar-10h/cifar-10-npy/test_labels.npy')
data = torch.from_numpy(original_images).float()
targets = torch.from_numpy(original_labels)
seed = 123
print(data.shape, targets.shape)
dataset = zip(data, targets)
randoms_labels = []
for i in range(100):
    dataset = zip(data, targets)
    random_label = tools.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, i+1)
    randoms_labels.append(random_label)
randoms_labels = np.array(randoms_labels)
print(randoms_labels.shape)
np.save(os.path.join('../data/cifar-10h/idn-100-users/test_idn20.npy'), randoms_labels)


