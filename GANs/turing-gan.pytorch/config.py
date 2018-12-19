# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Rahul Bhalley"

# Some configuration for networks
IMG_DIM = 32 # [32, 256]
if IMG_DIM == 32:   
    Z_DIM = 64
elif IMG_DIM == 256:
    Z_DIM = 100
BATCH_SIZE = 64
MODE = 'wgan' # [wgan, sgan]

# Some other configurations
DATASET = 'cifar-10' # [mnist, fashion-mnist, cifar-10, cifar-100, ...]
N_CHANNELS = 1 if DATASET in ['fashion-mnist', 'mnist'] else 3
BEGIN_ITER = 0
TOTAL_ITERS = 100000
ITERS_PER_LOG = 100
VERBOSE = True
TRAIN = True

print('------------------')
print('| Configurations |')
print('------------------')
print('')
print('IMG_DIM:         {}'.format(IMG_DIM))
print('Z_DIM:           {}'.format(Z_DIM))
print('BATCH_SIZE:      {}'.format(BATCH_SIZE))
print('MODE:            {}'.format(MODE))
print('DATASET:         {}'.format(DATASET))
print('N_CHANNELS:      {}'.format(N_CHANNELS))
print('BEGIN_ITER:      {}'.format(BEGIN_ITER))
print('TOTAL_ITERS:     {}'.format(TOTAL_ITERS))
print('ITERS_PER_LOG:   {}'.format(ITERS_PER_LOG))
print('VERBOSE:         {}'.format(VERBOSE))
print('')