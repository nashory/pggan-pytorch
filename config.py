""" config.py
"""
import argparse
import time

parser = argparse.ArgumentParser('PGGAN')

## general settings.
parser.add_argument('--train_data_root', type=str, default='/home1/irteam/nashory/data/CelebA/Img')
parser.add_argument('--random_seed', type=int, default=int(time.time()))
parser.add_argument('--n_gpu', type=int, default=1)             # for Multi-GPU training.






## training parameters.
parser.add_argument('--lr', type=float, default=0.001)          # learning rate.
parser.add_argument('--lr_decay', type=float, default=0.87)     # learning rate decay at every resolution transition.
parser.add_argument('--eps_drift', type=float, default=0.001)   # coeff for the drift loss.
parser.add_argument('--smoothing', type=float, default=0.997)   # smoothing factor for smoothed generator.
parser.add_argument('--nc', type=int, default=3)                # number of input channel.
parser.add_argument('--nz', type=int, default=512)              # input dimension of noise.
parser.add_argument('--ngf', type=int, default=512)             # feature dimension of final layer of generator.
parser.add_argument('--ndf', type=int, default=512)             # feature dimension of first layer of discriminator.
parser.add_argument('--TICK', type=int, default=1000)           # 1 tick = 1000 images = (1000/batch_size) iter.
parser.add_argument('--max_resl', type=int, default=8)          # 10-->1024, 9-->512, 8-->256
parser.add_argument('--trns_tick', type=int, default=200)       # transition tick
parser.add_argument('--stab_tick', type=int, default=100)       # stabilization tick


## network structure.
parser.add_argument('--flag_wn', type=bool, default=True)           # use of equalized-learning rate.
parser.add_argument('--flag_bn', type=bool, default=False)          # use of batch-normalization. (not recommended)
parser.add_argument('--flag_pixelwise', type=bool, default=True)    # use of pixelwise normalization for generator.
parser.add_argument('--flag_gdrop', type=bool, default=True)        # use of generalized dropout layer for discriminator.
parser.add_argument('--flag_leaky', type=bool, default=True)        # use of leaky relu instead of relu.
parser.add_argument('--flag_tanh', type=bool, default=False)        # use of tanh at the end of the generator.
parser.add_argument('--flag_sigmoid', type=bool, default=False)     # use of sigmoid at the end of the discriminator.
parser.add_argument('--flag_add_noise', type=bool, default=True)    # add noise to the real image(x)
parser.add_argument('--flag_norm_latent', type=bool, default=False) # pixelwise normalization of latent vector (z)
parser.add_argument('--flag_add_drift', type=bool, default=True)   # add drift loss




## optimizer setting.
parser.add_argument('--optimizer', type=str, default='adam')        # optimizer type.
parser.add_argument('--beta1', type=float, default=0.0)             # beta1 for adam.
parser.add_argument('--beta2', type=float, default=0.99)            # beta2 for adam.


## display and save setting.
parser.add_argument('--use_tb', type=bool, default=True)            # enable tensorboard visualization
parser.add_argument('--save_img_every', type=int, default=20)       # save images every specified iteration.
parser.add_argument('--display_tb_every', type=int, default=5)      # display progress every specified iteration.


## parse and save config.
config, _ = parser.parse_known_args()
