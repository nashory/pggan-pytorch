import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from custom_layers import fadein_layer, ConcatTable
import copy



# defined for code simplicity.
def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False):
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:      layers.append(nn.BatchNorm2d(c_out))
    if leaky:   layers.append(nn.LeakyReLU(0.2))
    else:       layer.append(nn.ReLU())
    return layers

def conv(c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:      layers.append(nn.BatchNorm2d(c_out))
    if leaky:   layers.append(nn.LeakyReLU(0.2))
    else:       layer.append(nn.ReLU())
    return nn.Sequential(*layers)


def copy_weights(from_module, to_module):
    print to_module
    for k, v in to_module.state_dict().iteritems():
        try:
            to_module[k] = from_module[k]
        except:
            print 'module name does not match!!!'
    return to_module

def get_module_names(model):
    names = []
    for key, val in model.state_dict().iteritems():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_leaky
        self.nz = config.nz
        self.ngf = config.ngf
        self.layer_name = None
        self.model = self.get_init_gen()

    def first_block(self):
        layers = []
        ndim = self.ngf
        layers = deconv(layers, self.nz, ndim, 4, 1, 0, self.flag_leaky, self.flag_bn)
        layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn)
        return  nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):
        halving = False
        layer_name = 'itermediate_{}x{}_{}x{}'.format(pow(2,resl-1), pow(2,resl-1), pow(2, resl), pow(2, resl))
        ndim = self.ngf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ngf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(resl-5):
                ndim = ndim/2
        layers = []
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))       # scale up by factor of 2.0
        if halving:
            layers = deconv(layers, ndim*2, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn)
        else:
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn)
        return  nn.Sequential(*layers), ndim, layer_name
    
    def to_rgb_block(self, c_in):
        layers = []
        layers.append(nn.ConvTranspose2d(c_in, 3, 1, 1, 0))
        if self.flag_tanh:  layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def get_init_gen(self):
        model = nn.Sequential()
        first_block, ndim = self.first_block()
        model.add_module('first_block', first_block)
        model.add_module('to_rgb_block', self.to_rgb_block(ndim))
        return model
    
    def grow_network(self, resl):
        # we make new network since pytorch does not support remove_module()
        new_model = nn.Sequential()
        names = get_module_names(self.model)
        for name, module in self.model.named_children():
            if not name=='to_rgb_block':
                new_model.add_module(name, module)                      # make new structure and,
                new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights
            
        if resl >= 3 and resl <= 9:
            print 'growing network[{}x{} to {}x{}]. It may take few seconds...'.format(pow(2,resl-1), pow(2,resl-1), pow(2,resl), pow(2,resl))
            low_resl_to_rgb = copy.deepcopy(self.model.to_rgb_block)     # make deep copy of the last block
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_upsample', nn.UpsamplingNearest2d(scale_factor=2))
            prev_block.add_module('low_resl_to_rgb', low_resl_to_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_block', inter_block)
            next_block.add_module('high_resl_to_rgb', self.to_rgb_block(ndim))

            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))
            self.model = new_model
           

    def flush_network(self):
        try:
            # make deep copy and paste.
            high_resl_block = copy.deepcopy(self.model.concat_block.layer2.high_resl_block)
            high_resl_to_rgb = copy.deepcopy(self.model.concat_block.layer2.high_resl_to_rgb)
           
            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                if name!='concat_block' and name!='fadein_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            # now, add the high resolution block.
            new_model.add_module(self.layer_name, high_resl_block)
            new_model.add_module('to_rgb_block', high_resl_to_rgb)
            self.model = new_model
        except:
            self.model = self.model
    
    def forward(self, x):
        # if fadein layer flag == True --> flush
        print 'forward'

        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flag_bn = False
        self.flag_pixelwise = False
        self.flag_wn = False
        self.flag_leakyrelu = True

    def first_block(self):
        print 'grow network'
    def intermediate_block(self):
        print 'grow network'
    def to_rgb_block(self):
        print 'grow network'
    
    def grow_network():
        print 'grow network'

    def flush_network():
        print 'flush network'
    def forward():
        print 'forward'

        








