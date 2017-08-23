# Copyright 2017 Queequeg92.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=10)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    print(len(dataset))
    for inputs, targets in dataloader:
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)

mean_cifar100 = (0.5071, 0.4866, 0.4409)
std_cifar100 = (0.2009, 0.1984, 0.2023)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.numerator = 0
        self.denominator = 0

    def update(self, val, n=1):
        self.val = val
        self.numerator += val
        self.denominator += n
        self.avg = float(self.numerator) / float(self.denominator)

if __name__ == '__main__':
    # Mean and std used to normalize cifar10.
    transform_train = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    mean, std = get_mean_and_std(trainset)
    print(mean) # output: (0.4914, 0.4822, 0.4465)
    print(std)  # output: (0.2023, 0.1994, 0.2010)

    # Mean and std used to normalize cifar100.
    transform_train = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    mean, std = get_mean_and_std(trainset)
    print(mean) # output: (0.5071, 0.4866, 0.4409)
    print(std)  # output: (0.2009, 0.1984, 0.2023)



