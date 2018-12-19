# -*- coding: utf-8 -*-

"""Wasserstein and Standard Turing GAN with Spectral Normalization
"""

from __future__ import print_function
from __future__ import division

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn
import torch.optim as optim

from config import *
if IMG_DIM == 32:
    from t_sn_gan_32 import *
elif IMG_DIM == 256:
    from t_sn_gan_256 import *

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import os

# Make experiments reproducible
_ = torch.manual_seed(12345)

####################
# Make directories #
# - Samples        #
# - Checkpoints    #
####################

if not os.path.exists(MODE):
    os.mkdir(MODE)
# Directory for samples
if not os.path.exists(os.path.join(MODE, 'samples')):
    os.mkdir(os.path.join(MODE, 'samples'))
if not os.path.exists(os.path.join(MODE, 'samples', DATASET)):
    os.mkdir(os.path.join(MODE, 'samples', DATASET))
# Directory for checkpoints
if not os.path.exists(os.path.join(MODE, 'checkpoints')):
    os.mkdir(os.path.join(MODE, 'checkpoints'))
if not os.path.exists(os.path.join(MODE, 'checkpoints', DATASET)):
    os.mkdir(os.path.join(MODE, 'checkpoints', DATASET))

####################
# Load the dataset #
####################

import psutil
cpu_cores = psutil.cpu_count()

transform = transforms.Compose(
    [
        transforms.Resize(IMG_DIM),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ],
)

root = '/Users/rahulbhalley/.torch/datasets/' + DATASET
if DATASET == 'cifar-10':
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
elif DATASET == 'cifar-100':
    trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
elif DATASET == 'mnist':
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
elif DATASET == 'fashion-mnist':
    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
else:
    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_cores)

def get_infinite_data(dataloader):
    while True:
        for images, _ in dataloader:
            yield images

data = get_infinite_data(dataloader)

################################################
# Define neural nets, losses, optimizers, etc. #
################################################

# Automatic GPU/CPU device placement
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create models
decode_model = Decoder().to(device)
encode_model = Encoder().to(device)
critic_model = Critic().to(device)

# Optimizers
decode_optim = optim.Adam(decode_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
encode_optim = optim.Adam(encode_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
critic_optim = optim.Adam(critic_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

############
# Training #
############

def train():
    print('Begin training!')
    # Try loading the latest existing checkpoints based on `BEGIN_ITER`
    try:
        # Checkpoints dirs
        decode_model_dir = os.path.join(MODE, 'checkpoints', DATASET, 'decode_model_' + str(BEGIN_ITER) + '.pth')
        encode_model_dir = os.path.join(MODE, 'checkpoints', DATASET, 'encode_model_' + str(BEGIN_ITER) + '.pth')
        critic_model_dir = os.path.join(MODE, 'checkpoints', DATASET, 'critic_model_' + str(BEGIN_ITER) + '.pth')
        # Load checkpoints
        decode_model.load_state_dict(torch.load(decode_model_dir, map_location='cpu'))
        encode_model.load_state_dict(torch.load(encode_model_dir, map_location='cpu'))
        critic_model.load_state_dict(torch.load(critic_model_dir, map_location='cpu'))
        print('Loaded the latest checkpoints from {}th iteration.')
        print('NOTE: Set the begin iteration in accordance to saved checkpoints.')
        # Free some memory
        del decode_model_dir, encode_model_dir, critic_model_dir
    except:
        print("Resume: Couldn't load the checkpoints from {}th iteration.".format(BEGIN_ITER))

    # Just to see the learning progress
    fixed_z = torch.randn(BATCH_SIZE * 2, Z_DIM, 1, 1).to(device)

    for i in range(BEGIN_ITER, TOTAL_ITERS+1):
        # Just because I'm encountering some problem with
        # the batch size of sampled data with `torchvision`.
        def safe_sampling():
            x_sample = data.next()
            if x_sample.size(0) != BATCH_SIZE:
                print('Required batch size not equal to x_sample batch size: {} != {} | skipping...'.format(BATCH_SIZE, x_sample.size(0)))
                x_sample = data.next()
            return x_sample.to(device)

        ######################
        # Train critic_model #
        # Train encode_model #
        ######################
        
        # Gradient computation shut down:
        # - decode_model
        
        for param in critic_model.parameters():
            param.requires_grad_(True)
        for param in encode_model.parameters():
            param.requires_grad_(True)
        for param in decode_model.parameters():
            param.requires_grad_(False)

        for j in range(1):
            z_sample = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device) # Sample prior from Gaussian distribution
            x_sample = safe_sampling()
            with torch.no_grad():
                x_fake = decode_model(z_sample)
            x_real_encoded = encode_model(x_sample)
            x_fake_encoded = encode_model(x_fake)
            x_real_fake = x_real_encoded - x_fake_encoded
            x_fake_real = x_fake_encoded - x_real_encoded
            x_real_fake_score = critic_model(x_real_fake)
            x_fake_real_score = critic_model(x_fake_real)

            # Compute loss for critic, encoder
            # Compute gradients
            critic_optim.zero_grad()
            encode_optim.zero_grad()
            
            # Decide distance for loss
            if MODE == 'wgan':
                d_loss = - (x_real_fake_score - x_fake_real_score)
            elif MODE == 'sgan':
                d_loss = - torch.log(x_real_fake_score) - torch.log(1 - x_fake_real_score)

            d_loss = d_loss.mean()
            d_loss.backward()

            # Update encode_model & critic_model
            critic_optim.step()
            encode_optim.step()

        ######################
        # Train decode_model #
        ######################

        # Gradient computation shut down:
        # - critic_model
        # - encode_model

        for param in critic_model.parameters():
            param.requires_grad_(False)
        for param in encode_model.parameters():
            param.requires_grad_(False)
        for param in decode_model.parameters():
            param.requires_grad_(True)

        for j in range(2):
            z_sample = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device) # Sample prior from Gaussian distribution
            x_sample = safe_sampling()
            
            x_fake = decode_model(z_sample)
            x_real_encoded = encode_model(x_sample)
            x_fake_encoded = encode_model(x_fake)
            x_real_fake = x_real_encoded - x_fake_encoded
            x_fake_real = x_fake_encoded - x_real_encoded
            x_real_fake_score = critic_model(x_real_fake)
            x_fake_real_score = critic_model(x_fake_real)

            # Compute loss for decoder
            # Compute gradients
            decode_optim.zero_grad()

            # Decide distance for loss
            if MODE == 'wgan':
                g_loss = - (x_fake_real_score - x_real_fake_score)
            elif MODE == 'sgan':
                g_loss = - torch.log(1 - x_real_fake_score) - torch.log(x_fake_real_score)
            
            g_loss = g_loss.mean()
            g_loss.backward()

            # Update decode_model
            decode_optim.step()
        
        ##################
        # Log statistics #
        ##################

        if i % ITERS_PER_LOG == 0:
            # Print statistics
            print('iter: {}, d_loss: {}, g_loss: {}'.format(i, d_loss, g_loss))
            # Save image grids of fake and real images
            with torch.no_grad():
                #z_sample = torch.randn(BATCH_SIZE * 2, Z_DIM, 1, 1)
                samples = decode_model(fixed_z)
            samples_dir = os.path.join(MODE, 'samples', DATASET, 'test_{}.png'.format(i))
            real_samples_dir = os.path.join(MODE, 'samples', DATASET, 'real.png')
            vutils.save_image(samples, samples_dir, normalize=True)
            vutils.save_image(x_sample, real_samples_dir, normalize=True)
            # Checkpoint directories
            decode_model_dir = os.path.join(MODE, 'checkpoints', DATASET, 'decode_model_' + str(i) + '.pth')
            encode_model_dir = os.path.join(MODE, 'checkpoints', DATASET, 'encode_model_' + str(i) + '.pth')
            critic_model_dir = os.path.join(MODE, 'checkpoints', DATASET, 'critic_model_' + str(i) + '.pth')
            # Save all the checkpoints
            torch.save(decode_model.state_dict(), decode_model_dir)
            torch.save(encode_model.state_dict(), encode_model_dir)
            torch.save(critic_model.state_dict(), critic_model_dir)
            # Free some memory
            del decode_model_dir, encode_model_dir, critic_model_dir
            
    print('Finished training!')


def infer(n=1, epoch=100000):
    try:
        decode_model_dir = os.path.join(MODE, 'checkpoints', DATASET, 'decode_model_' + str(epoch) + '.pth')
        decode_model.load_state_dict(torch.load(decode_model_dir, map_location='cpu'))
    except:
        print('Could not load checkpoint of `decode_model`.')

    for i in range(n):
        with torch.no_grad():
            z_sample = torch.randn(BATCH_SIZE * 2, Z_DIM, 1, 1)
            samples = decode_model(z_sample)
        samples_dir = os.path.join(MODE, 'samples', DATASET, 'latest_{}.png'.format(i))
        vutils.save_image(samples, samples_dir, normalize=True)
        print('Saved image: {}'.format(samples_dir))

# Train the Turing GAN
train()
# Sample for Turing GAN
#infer()
