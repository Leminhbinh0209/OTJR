from __future__ import print_function
import sys
import os
import socket
import numpy as np
from typing import Any, Callable, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
sys.path.append('../')


def get_intel_dataloaders(config):
    """
    Intel data loader
    
    """
    def worker_init_fn(worker_id):                                                          
        np.random.seed(config.SYS.random_seed + worker_id)

    # No normalization to perform the adversarial attack
    train_transform = transforms.Compose([
            transforms.Resize((config.MODEL.input_size,config.MODEL.input_size)),
            transforms.RandomRotation(12),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.Resize((config.MODEL.input_size,config.MODEL.input_size)),
        transforms.ToTensor(),
        ])

    train_set  = datasets.ImageFolder(config.TRAIN.data_root, train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=config.TRAIN.batch_size,
                              shuffle=True,
                              num_workers=config.SYS.num_workers,
                              pin_memory=False,
                              worker_init_fn=worker_init_fn)

    # Define test loader
    test_set = datasets.ImageFolder(config.TEST.data_root, test_transform)
    test_loader = DataLoader(test_set, 
                             batch_size=config.TEST.batch_size,
                             shuffle=False,
                             num_workers=config.SYS.num_workers, 
                             pin_memory=False,
                             worker_init_fn=worker_init_fn)

    return train_loader, test_loader