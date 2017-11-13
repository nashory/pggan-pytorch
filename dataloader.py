import os
import torch
import numpy as np
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt



class dataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        self.batch_table = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:4, 1024:1} # change this according to available gpu memory.

    def renew(self, config):
        if config.num_workers==None:    config.num_workers = 8
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))
        dataset = ImageFolder(
                    root=self.root,
                    transform=transforms.Compose([
                        transforms.Scale(size=config.image_size),
                        transforms.ToTensor(),]))       

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        return dataloader
    
    def get_batch(self, config):
       print 'get batch.' 
        

        









