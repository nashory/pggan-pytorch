""" config.py
"""

import argparse

parser = argparse.ArgumentParser('PGGAN')

## general settings.
parser.add_argument('--train_data_root', type=str, default='/home1/work/nashory/data/CelebA/Img')
parser.add_argument('--random_seed', type=int, default=12345)



## training parameters.
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--nz', type=int, default=512)
parser.add_argument('--ngf', type=int, default=512)
parser.add_argument('--ndf', type=int, default=512)
parser.add_argument('--TICK', type=int, default=1000)           # 1 tick = 1000 images = (1000/batch_size) iter.
parser.add_argument('--max_res', type=int, default=10)          # 10 --> 1024
parser.add_argument('--trns_tick', type=int, default=40)        # transition tick
parser.add_argument('--stab_tick', type=int, default=40)        # stabilization tick


## network structure.
parser.add_argument('--flag_wn', type=bool, default=False)
parser.add_argument('--flag_bn', type=bool, default=False)
parser.add_argument('--flag_pixelwise', type=bool, default=False)
parser.add_argument('--flag_leaky', type=bool, default=True)
parser.add_argument('--flag_tanh', type=bool, default=True)



## optimizer setting.
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.999)




## display and save setting.
parser.add_argument('--use_tensorboard', type=bool, default=True)





parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--z_num', type=int, default=64)
parser.add_argument('--n_num', type=int, default=64)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--is_train', type=bool, default=True)
parser.add_argument('--start_step', type=int, default=1)
parser.add_argument('--final_step', type=int, default=500000)
parser.add_argument('--save_step', type=int, default=1000)
parser.add_argument('--log_step', type=int, default=200)
parser.add_argument('--log_path', type=str, default='logdir')
parser.add_argument('--image_path', type=str, default='images')
parser.add_argument('--model_path', type=str, default='checkpoint')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--load_step', type=int, default=0)


## parse and save config.
config, _ = parser.parse_known_args()
