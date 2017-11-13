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

# same function as ConcatTable container in Torch7.
class ConcatTable(nn.Module):
    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        
    def forward(self,x):
        y = [self.layer1(x), self.layer2(x)]
        return y

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class fadein_layer(nn.Module):
    def __init__(self, config):
        super(fadein_layer, self).__init__()
        self.alpha = 0.0
        self.complete = 0.0
        self.phase = 'gtrns'
        self.TICK = config.TICK
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.delta = 1.0/self.TICK
        self.flag_flush = False
    
    def update_alpha(self, batch_size):
        self.alpha = self.alpha + batch_size*(self.delta/self.trns_tick)
        self.alpha = max(0, min(self.alpha, 1.0))
        self.complete = self.complete + batch_size*(self.delta/(self.trns_tick+self.stab_tick))
        self.complete = max(0, min(self.complete, 1.0))
        if self.alpha == 1.0:
            self.phase = 'gstab'
        if self.complete == 1.0:
             self.flag_flush = True


    def forward(self, x_low, x_high):
        return torch.add(x_low.mul(1.0-self.alpha), x_high.mul(self.alpha))


class minibatch_std_concat_layer(nn.Module):
    def __init__(self, config):
        print 'minibatch std concat layer'






