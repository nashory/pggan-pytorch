""" config.py
"""

import argparse

parser = argparse.ArgumentParser('PGGAN')
parser.add_argument('--dataset', type=str, default='/home1/irteam/work/nashory/data')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--z_num', type=int, default=64)
parser.add_argument('--n_num', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--random_seed', type=int, default=12345)
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
parser.add_argument('--use_tensorboard', type=bool, default=True)


def get_config():
    config, _ = parser.parse_known_args()
    return config, _
