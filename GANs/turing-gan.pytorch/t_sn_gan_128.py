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
            SpectralNorm(nn.Conv2d(N_CHANNELS, IMG_DIM, 6, 2, padding=2)),
            nn.LeakyReLU()
        ]

        block2 = [
            SpectralNorm(nn.Conv2d(IMG_DIM, IMG_DIM * 2, 6, 2, padding=2)),
            nn.BatchNorm2d(IMG_DIM * 2),
            nn.LeakyReLU()
        ]

        block3 = [
            SpectralNorm(nn.Conv2d(IMG_DIM * 2, IMG_DIM * 4, 6, 2, padding=2)),
            nn.BatchNorm2d(IMG_DIM * 4),
            nn.LeakyReLU()
        ]

        block4 = [
            SpectralNorm(nn.Conv2d(IMG_DIM * 4, IMG_DIM * 8, 6, 2, padding=2)),
            nn.BatchNorm2d(IMG_DIM * 8),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        ]

        all_blocks = block1 + block2 + block3 + block4
        self.main = nn.Sequential(*all_blocks)
        
        # Free some memory
        del all_blocks, block1, block2, block3, block4

        # Print summary if VERBOSE is True
        #if VERBOSE:
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
                SpectralNorm(nn.Linear(1024, 512)),
                nn.LeakyReLU(),
                SpectralNorm(nn.Linear(512, 1, bias=False)),
                nn.Sigmoid()
            )
        elif MODE == 'wgan':
            self.main = nn.Sequential(
                SpectralNorm(nn.Linear(1024, 512)),
                nn.LeakyReLU(),
                SpectralNorm(nn.Linear(512, 1, bias=False))
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
        x = torch.zeros(BATCH_SIZE, 1024)
        
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
            nn.Linear(Z_DIM, 4 * 4 * IMG_DIM * 8),
            nn.BatchNorm1d(4 * 4 * IMG_DIM * 8),
            nn.ReLU()
        ]

        block2 = [
            nn.ConvTranspose2d(IMG_DIM * 8, IMG_DIM * 4, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(IMG_DIM * 4),
            nn.ReLU()
        ]

        block3 = [
            nn.ConvTranspose2d(IMG_DIM * 4, IMG_DIM * 2, 6, 2, padding=2),
            nn.BatchNorm2d(IMG_DIM * 2),
            nn.ReLU()
        ]

        block4 = [
            nn.ConvTranspose2d(IMG_DIM * 2, IMG_DIM, 6, 2, padding=2),
            nn.BatchNorm2d(IMG_DIM),
            nn.ReLU()
        ]

        block5 = [
            nn.ConvTranspose2d(IMG_DIM, IMG_DIM // 2, 6, 2, padding=2),
            nn.BatchNorm2d(IMG_DIM // 2),
            nn.ReLU()
        ]

        block6 = [
            nn.ConvTranspose2d(IMG_DIM // 2, N_CHANNELS, 6, 2, padding=2),
            nn.Tanh()
        ]

        all_blocks = block2 + block3 + block4 + block5 + block6
        self.main1 = nn.Sequential(*block1)
        self.main2 = nn.Sequential(*all_blocks)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, block6
        
        # Print summary if VERBOSE is True
        if VERBOSE:
            self.summary()

    def forward(self, x):
        x = x.squeeze(-2).squeeze(-1)
        x = self.main1(x)
        x = x.view(-1, IMG_DIM * 8, 4, 4)
        x = self.main2(x)
        return x

    def summary(self):
        x = torch.zeros(BATCH_SIZE, Z_DIM, 1, 1)
        x = x.squeeze(-2).squeeze(-1)
        
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
            for layer in self.main1:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            #x = x.view(-1, IMG_DIM * 8, 4, 4)
            x = torch.randn(BATCH_SIZE, IMG_DIM * 8, 4, 4)
            for layer in self.main2:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))