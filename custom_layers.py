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

    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))

    # input : [x_low, x_high] from ConcatTable()
    def forward(self, x):
        return torch.add(x[0].mul(1.0-self.alpha), x[1].mul(self.alpha))


class minibatch_std_concat_layer(nn.Module):
    def __init__(self):
        super(minibatch_std_concat_layer, self).__init__()
        
    def forward(self, x):
        std = x.clone().std(1, keepdim=True)
        return torch.cat((x, std), 1)


class pixelwise_norm_layer(nn.Module):
    def __init__(self):
        super(pixelwise_norm_layer, self).__init__()
        self.elipson = 1e-8

    def forward(self, x):
        nf = x.size(1)
        t = x.clone()
        t.data.fill_(0)
        norm = torch.sqrt(x.pow(2).sum(1, keepdim=True).expand_as(x).div(nf).add(self.elipson))
        return torch.addcdiv(t, 1, x, norm)


# for equaliaeed-learning rate.
class equalized_conv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad)
        if initializer == 'kaiming':    torch.nn.init.kaiming_normal(self.conv.weight)
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.conv.weight)
        self.inv_c = np.sqrt(2.0/(c_in*k_size**2))

    def forward(self, x):
        return self.conv(x.mul(self.inv_c))
        
 
class equalized_deconv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'):
        super(equalized_deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad)
        if initializer == 'kaiming':    torch.nn.init.kaiming_normal(self.deconv.weight)
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.deconv.weight)
        self.inv_c = np.sqrt(2.0/(c_in*k_size**2))

    def forward(self, x):
        return self.deconv(x.mul(self.inv_c))


class equalized_linear(nn.Module):
    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out)
        if initializer == 'kaiming':    torch.nn.init.kaiming_normal(self.linear.weight)
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.linear.weight)
        self.inv_c = np.sqrt(2.0/(c_in))

    def forward(self, x):
        return self.linear(x.mul(self.inv_c))
