from __future__ import print_function
import sys
sys.path.append('../')
import os
import socket
import numpy as np
from typing import Any, Callable, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# reference from https://github.com/HobbitLong/RepDistiller/blob/master/dataset/cifar100.py

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar10_dataloaders(config):
    """
    cifar 10 data loader
    
    """
    def worker_init_fn(worker_id):                                                          
        np.random.seed(config.SYS.random_seed + worker_id)

    drop_last = False


    train_transform = transforms.Compose([
        transforms.RandomCrop(config.MODEL.input_size, padding=4),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.CIFAR10(root=f"{config.TRAIN.data_root}/{config.dataset}/",
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=config.TRAIN.batch_size,
                              shuffle=True,
                              num_workers=config.SYS.num_workers,
                              pin_memory=False,
                              drop_last=drop_last,
                              worker_init_fn=worker_init_fn)

    # Define test loader
    test_set = datasets.CIFAR10(root=f"{config.TRAIN.data_root}/{config.dataset}/",
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,       
                             batch_size=config.TEST.batch_size,
                             shuffle=False,
                             num_workers=config.SYS.num_workers, 
                             pin_memory=False,
                             worker_init_fn=worker_init_fn)

    return train_loader, test_loader

def get_cifar100_dataloaders(config):
    """
    cifar 10 data loader
    
    """
    def worker_init_fn(worker_id):                                                          
        np.random.seed(config.SYS.random_seed + worker_id)

    drop_last = False


    train_transform = transforms.Compose([
        transforms.RandomCrop(config.MODEL.input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.CIFAR100(root=f"{config.TRAIN.data_root}/{config.dataset}/",
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=config.TRAIN.batch_size,
                              shuffle=True,
                              num_workers=config.SYS.num_workers,
                              pin_memory=False,
                              drop_last=drop_last,
                              worker_init_fn=worker_init_fn)

    # Define test loader
    test_set = datasets.CIFAR100(root=f"{config.TRAIN.data_root}/{config.dataset}/",
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,       
                             batch_size=config.TEST.batch_size,
                             shuffle=False,
                             num_workers=config.SYS.num_workers, 
                             pin_memory=False,
                             worker_init_fn=worker_init_fn)

    return train_loader, test_loader
