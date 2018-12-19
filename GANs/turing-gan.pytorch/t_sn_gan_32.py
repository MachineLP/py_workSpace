# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn

from spectral_normalization import SpectralNorm
from config import *

#############################
# Neural Networks           #
#############################
# ------------------------- #
# Encoder                   #
# ------------------------- #
# Discriminator/Critic      #
# ------------------------- #
# Decoder/Builder/Generator #
# ------------------------- #
#############################


###########
# Encoder #
###########

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        block1 = [
            SpectralNorm(nn.Conv2d(N_CHANNELS, 32, 5, 1)),
            SpectralNorm(nn.BatchNorm2d(32)),
            nn.LeakyReLU()
        ]

        block2 = [
            SpectralNorm(nn.Conv2d(32, 64, 4, 2)),
            SpectralNorm(nn.BatchNorm2d(64)),
            nn.LeakyReLU()
        ]

        block3 = [
            SpectralNorm(nn.Conv2d(64, 128, 4, 1)),
            SpectralNorm(nn.BatchNorm2d(128)),
            nn.LeakyReLU()
        ]

        block4 = [
            SpectralNorm(nn.Conv2d(128, 256, 4, 2)),
            SpectralNorm(nn.BatchNorm2d(256)),
            nn.LeakyReLU()
        ]

        block5 = [
            SpectralNorm(nn.Conv2d(256, Z_DIM, 1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))    # Global average pooling in 2d
        ]

        all_blocks = block1 + block2 + block3 + block4 + block5
        self.main = nn.Sequential(*all_blocks)

        # Free some memory
        del block1, block2, block3, block4, block5, all_blocks

        # Print summary if VERBOSE is True
        if VERBOSE:
            self.summary()
    
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

    def summary(self):
        x = torch.zeros(BATCH_SIZE, N_CHANNELS, IMG_DIM, IMG_DIM)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            for layer in self.main:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))


########################
# Discriminator/Critic #
########################

class Critic(nn.Module):
    
    def __init__(self):
        super(Critic, self).__init__()
        
        if MODE == 'sgan':
            self.main = nn.Sequential(
                SpectralNorm(nn.Linear(64, 256)),
                nn.LeakyReLU(0.1),
                SpectralNorm(nn.Linear(256, 1, bias=False)),
                nn.Sigmoid()
            )
        elif MODE == 'wgan':
            self.main = nn.Sequential(
                SpectralNorm(nn.Linear(64, 256)),
                nn.LeakyReLU(0.1),
                SpectralNorm(nn.Linear(256, 1, bias=False))
            )
        
        # Print summary if VERBOSE is True
        if VERBOSE:
            self.summary()
    
    def forward(self, x):
        x = x.squeeze(-2).squeeze(-1)   # I think encoded fake & real images [BATCH_SIZE, 64, 1, 1] 
                                        # are fed in Critic and therefore must be reshaped
                                        # to [BATCH_SIZE, 64].
        x = self.main(x)
        return x

    def summary(self):
        x = torch.zeros(BATCH_SIZE, 64)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            for layer in self.main:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))


#############################
# Decoder/Builder/Generator #
#############################

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        block1 = [
            nn.ConvTranspose2d(Z_DIM, 256, 4, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        ]

        block2 = [
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        ]

        block3 = [
            nn.ConvTranspose2d(128, 64, 4, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        ]

        block4 = [
            nn.ConvTranspose2d(64, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        ]

        block5 = [
            nn.ConvTranspose2d(32, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        ]

        block6 = [
            nn.Conv2d(32, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, N_CHANNELS, 1, 1),
            nn.Tanh()
        ]

        all_blocks = block1 + block2 + block3 + block4 + block5 + block6
        self.main = nn.Sequential(*all_blocks)

        # Free some memory
        del block1, block2, block3, block4, block5, all_blocks

        # Print summary if VERBOSE is True
        if VERBOSE:
            self.summary()

    def forward(self, x):
        #x = x.view(-1, 1, 1, Z_DIM)
        for layer in self.main:
            x = layer(x)
        return x

    def summary(self):
        x = torch.zeros(BATCH_SIZE, Z_DIM, 1, 1)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            for layer in self.main:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))