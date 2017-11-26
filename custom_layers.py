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
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming', bias=False):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    torch.nn.init.kaiming_normal(self.conv.weight)
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.conv.weight)
        
        conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = np.sqrt(conv_w.pow(2).mean()+1e-8)
        inv_w = conv_w.clone().fill_(self.scale)
        t = inv_w.clone().fill_(0)
        self.conv.weight.data = torch.addcdiv(t, 1, self.conv.weight.data, inv_w)            # adjust weights dynamically.

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1,-1,1,1).expand_as(x)
        
 
class equalized_deconv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'):
        super(equalized_deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    torch.nn.init.kaiming_normal(self.deconv.weight)
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.deconv.weight)
        
        deconv_w = self.deconv.weight.data.clone()
        self.scale = np.sqrt(deconv_w.pow(2).mean()+1e-8)
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        inv_w = deconv_w.clone().fill_(self.scale)
        t = inv_w.clone().fill_(0)
        self.deconv.weight.data = torch.addcdiv(t, 1, self.deconv.weight.data, inv_w)            # adjust weights dynamically.

    def forward(self, x):
        x = self.deconv(x.mul(self.scale))
        return x + self.bias.view(1,-1,1,1).expand_as(x)


class equalized_linear(nn.Module):
    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':    torch.nn.init.kaiming_normal(self.linear.weight)
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.linear.weight)
        
        linear_w = self.linear.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = np.sqrt(linear_w.pow(2).mean()+1e-8)
        inv_w = linear_w.clone().fill_(self.scale)
        t = inv_w.clone().fill_(0)
        self.linear.weight.data = torch.addcdiv(t, 1, self.linear.weight.data, inv_w)            # adjust weights dynamically.

    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1,-1).expand_as(x)






