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
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5


# for equaliaeed-learning rate.
class equalized_conv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming', bias=False):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    torch.nn.init.kaiming_normal(self.conv.weight)
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.conv.weight)
        
        conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
<<<<<<< HEAD
        self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data/self.scale)
=======
        self.scale = np.sqrt(conv_w.pow(2).mean())
        inv_w = conv_w.clone().fill_(self.scale)
        t = inv_w.clone().fill_(0)
        self.conv.weight.data = torch.addcdiv(t, 1, self.conv.weight.data, inv_w)            # adjust weights dynamically.
>>>>>>> 15d4fa8a8bdba6078f1a98296a57ba6825a754a5

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
<<<<<<< HEAD
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.deconv.weight.data ** 2)) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data/self.scale)
        
=======
        self.scale = np.sqrt(deconv_w.pow(2).mean())
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        inv_w = deconv_w.clone().fill_(self.scale)
        t = inv_w.clone().fill_(0)
        self.deconv.weight.data = torch.addcdiv(t, 1, self.deconv.weight.data, inv_w)            # adjust weights dynamically.

>>>>>>> 15d4fa8a8bdba6078f1a98296a57ba6825a754a5
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
<<<<<<< HEAD
        self.scale = (torch.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data/self.scale)
        
=======
        self.scale = np.sqrt(linear_w.pow(2).mean())
        inv_w = linear_w.clone().fill_(self.scale)
        t = inv_w.clone().fill_(0)
        self.linear.weight.data = torch.addcdiv(t, 1, self.linear.weight.data, inv_w)            # adjust weights dynamically.

>>>>>>> 15d4fa8a8bdba6078f1a98296a57ba6825a754a5
    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1,-1).expand_as(x)


<<<<<<< HEAD
# ref: https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class generalized_drop_out(nn.Module):
    def __init__(self, mode='mul', strength=0.4, axes=(0,1), normalize=False):
        super(generalized_drop_out, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str



=======
>>>>>>> 15d4fa8a8bdba6078f1a98296a57ba6825a754a5




