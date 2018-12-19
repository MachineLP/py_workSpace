# Turing Generative Adversarial Network
[Turing GANs](https://arxiv.org/abs/1810.10948) are *quick to train!* This excited me to write my own versions in [PyTorch](https://pytorch.org) which is based on the original Keras [implementation](https://github.com/bojone/).

Thanks [Jianlin Su](https://github.com/bojone/T-GANs), *creator of Turing GANs*, for suggesting this code as [PyTorch](https://pytorch.org) implementation of [Turing GANs](https://arxiv.org/abs/1810.10948)!

## Experiments
So, following are my experiments' resulting image data.

### Note
- For all the experiments the images shown below are sampled after **100K iterations** of training the Turing GAN on various datasets. 
- All the experiments used [spectral normalization](https://arxiv.org/abs/1802.05957) for 1-Lipschitz contraint enforcement. 
- I trained all of the Turing GANs with both *Jensen-Shannon* and *Wasserstein* divergences.
- All experiments were performed with same hyper-parameters as devised in paper.

Using [32-sized Turing GAN](https://github.com/rahulbhalley/turing-gan.pytorch/blob/master/t_sn_gan_32.py) I performed experiments on the following dataset(s):
- CIFAR-10
- MNIST
- Fashion-MNIST

### CIFAR-10
#### Turing Standard GAN (Left) | Turing Wasserstein GAN (Right) [Both Spectrally Normalized]
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/sgan/samples/cifar-10/latest_100000.png)
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/wgan/samples/cifar-10/latest_100000.png)

### MNIST
#### Turing Standard GAN (Left) | Turing Wasserstein GAN (Right) [Both Spectrally Normalized]
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/sgan/samples/mnist/latest_100000.png)
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/wgan/samples/mnist/latest_100000.png)

### Fashion MNIST
#### Turing Standard GAN (Left) | Turing Wasserstein GAN (Right) [Both Spectrally Normalized]
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/sgan/samples/fashion-mnist/latest_100000.png)
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/wgan/samples/fashion-mnist/latest_100000.png)

## References
- Training Generative Adversarial Networks Via Turing Test [[arXiv](https://arxiv.org/abs/1810.10948)]
- Original [T-GANs](https://github.com/bojone/T-GANs) implementation
- Spectral Normalization for Generative Adversarial Networks [[arXiv](https://arxiv.org/abs/1802.05957)]
- Spectral Normalization [implementation](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py) in [PyTorch](https://pytorch.org)

## Contact
Reach me at `rahulbhalley@icloud.com`.
