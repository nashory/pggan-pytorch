# generate interpolated images.


import os,sys
import torch
from config import config
from torch.autograd import Variable
import utils as utils


use_cuda = True
checkpoint_path = 'repo/model/gen_R8_T55.pth.tar'
n_intp = 20

# load trained model.
import network as net
test_model = net.Generator(config)
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    test_model = torch.nn.DataParallel(test_model).cuda(device=0)
else:
    torch.set_default_tensor_type('torch.FloatTensor')

for resl in range(3, config.max_resl+1):
    test_model.module.grow_network(resl)
    test_model.module.flush_network()
print(test_model)

print('load checkpoint form ... {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path)
test_model.module.load_state_dict(checkpoint['state_dict'])

# create folder.
for i in range(1000):
    name = 'repo/interpolation/try_{}'.format(i)
    if not os.path.exists(name):
        os.system('mkdir -p {}'.format(name))
        break;

# interpolate between twe noise(z1, z2).
z_intp = torch.FloatTensor(1, config.nz)
z1 = torch.FloatTensor(1, config.nz).normal_(0.0, 1.0)
z2 = torch.FloatTensor(1, config.nz).normal_(0.0, 1.0)
if use_cuda:
    z_intp = z_intp.cuda()
    z1 = z1.cuda()
    z2 = z2.cuda()
    test_model = test_model.cuda()

z_intp = Variable(z_intp)

for i in range(1, n_intp+1):
    alpha = 1.0/float(n_intp+1)
    z_intp.data = z1.mul_(alpha) + z2.mul_(1.0-alpha)
    fake_im = test_model.module(z_intp)
    fname = os.path.join(name, '_intp{}.jpg'.format(i))
    utils.save_image_single(fake_im.data, fname, imsize=pow(2,config.max_resl))
    print('saved {}-th interpolated image ...'.format(i))


