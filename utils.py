""" utils.py
"""

import os
import torch
import numpy as np
from io import BytesIO
import scipy.misc
import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt

def get_loader(config):
    root = os.path.join(os.path.abspath(os.curdir), config.dataset)
    print('[*] Load data from {0}.'.format(root))
    dataset = ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(size=config.image_size),
            transforms.ToTensor(),
        ])
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    return dataloader


def make_path(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.image_path):
        os.makedirs(config.image_path)

def load_model(net, path):
    """ load model
    """
    net.load_state_dict(torch.load(path))

def save_model(net, path):
    """ save model
    """
    torch.save(net.state_dict(), path)

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False):
    from PIL import Image
    tensor = tensor.cpu()
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, range=range, scale_each=True)

    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def make_summary(writer, key, value, step):
    if hasattr(value, '__len__'):
        for idx, img in enumerate(value):
            summary = tf.Summary()
            sio = BytesIO()
            scipy.misc.toimage(img).save(sio, format='png')
            image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
            summary.value.add(tag="{}/{}".format(key, idx), image=image_summary)
            writer.add_summary(summary, global_step=step)
    else:
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, global_step=step)
