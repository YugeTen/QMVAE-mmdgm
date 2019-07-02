from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from numpy import prod

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchnet.dataset import TensorDataset, ResampleDataset

svhn_dataSize = torch.Size([3, 32, 32])
svhn_imgChans = svhn_dataSize[0]
svhn_fBase = 32

data_size = torch.Size([1, 28, 28])
data_dim = int(prod(data_size))
hidden_dim = 400

eta = 1e-6

class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, expert):
        super(MVAE, self).__init__()
        self.mnist_enc = MNISTEncoder(n_latents)
        self.mnist_dec = MNISTDecoder(n_latents)
        self.svhn_enc  = SVHNEncoder(n_latents)
        self.svhn_dec  = SVHNDecoder(n_latents)
        self.experts       = MixtureOfExperts() if expert=="moe" else ProductOfExperts()
        self.n_latents     = n_latents

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        if not (os.path.exists('data/train-ms-mnist-idx.pt')
                and os.path.exists('data/train-ms-svhn-idx.pt')
                and os.path.exists('data/test-ms-mnist-idx.pt')
                and os.path.exists('data/test-ms-svhn-idx.pt')):
            raise RuntimeError('Transformed indices not found.'
                               + ' You can generate them with the script in bin')
        # get transformed indices
        t_mnist = torch.load('data/train-ms-mnist-idx.pt')
        t_svhn = torch.load('data/train-ms-svhn-idx.pt')
        s_mnist = torch.load('data/test-ms-mnist-idx.pt')
        s_svhn = torch.load('data/test-ms-svhn-idx.pt')
        # load base datasets
        t1, s1 = self.mnist_enc.getDataLoaders(batch_size, shuffle=shuffle, device=device)
        t2, s2 = self.svhn_enc.getDataLoaders(batch_size, shuffle=shuffle, device=device)
        # build resampleDataset instances and combine
        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
        ])
        test_mnist_svhn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = torch.utils.data.DataLoader(
            train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs
        )
        return train_loader, test_loader


    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, image=None, text=None):
        mu, logvar = self.infer(image, text)
        # reparametrization trick to sample
        z          = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = self.mnist_dec(z)
        txt_recon  = self.svhn_dec(z)
        return img_recon, txt_recon, mu, logvar

    def infer(self, mnist=None, svhn=None):
        batch_size = mnist.size(0) if mnist is not None else svhn.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)
        if mnist is not None:
            img_mu, img_logvar = self.mnist_enc(mnist)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if svhn is not None:
            txt_mu, txt_logvar = self.svhn_enc(svhn)
            mu     = torch.cat((mu, txt_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, txt_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))

class MNISTEncoder(nn.Module):

    """ Generate latent parameters for MNIST image data. """
    def __init__(self, n_latents):
        super(MNISTEncoder, self).__init__()
        self.fc1   = nn.Linear(784, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc31  = nn.Linear(512, n_latents)
        self.fc32  = nn.Linear(512, n_latents)
        self.swish = Swish()

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        # this is required if using the relaxedBernoulli because it doesn't
        # handle scoring values that are actually 0. or 1.
        tx = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda p: p.clamp(eta, 1 - eta))
        ])
        train_loader = DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class MNISTDecoder(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """
    def __init__(self, n_latents):
        super(MNISTDecoder, self).__init__()
        self.fc1   = nn.Linear(n_latents, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)
        self.fc4   = nn.Linear(512, 784)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)  # NOTE: no sigmoid here. See train.py


class SVHNEncoder(nn.Module):
    """ Generate latent parameters for SVHN image data. """
    def __init__(self, latent_dim):
        super(SVHNEncoder, self).__init__()
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(svhn_imgChans, svhn_fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(svhn_fBase, svhn_fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(svhn_fBase * 2, svhn_fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(svhn_fBase * 4, latent_dim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(svhn_fBase * 4, latent_dim, 4, 1, 0, bias=False)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = DataLoader(
            datasets.SVHN('data', split='train', download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(
            datasets.SVHN('data', split='test', download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def forward(self, x):
        e = self.enc(x)
        return self.c1(e).squeeze(), F.softplus(self.c2(e)).squeeze() + eta


class SVHNDecoder(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """
    def __init__(self, latent_dim):
        super(SVHNDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, svhn_fBase * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(svhn_fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(svhn_fBase * 4, svhn_fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(svhn_fBase * 2, svhn_fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(svhn_fBase, svhn_imgChans, 4, 2, 1, bias=False),
            # Output size: 3 x 32 x 32 (no sigmoid)
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        return out  # Output size: 3 x 32 x 32


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M+1 x B x D for M experts
    @param logvar: M+1 x B x D for M experts
    (row 0 is always all zeros)
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

class MixtureOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M+1 x B x D for M experts
    @param logvar: M+1 x B x D for M experts
    (row 0 is always all zeros)
    """
    def forward(self, mu, logvar, eps=1e-8):
        if mu.shape[0] == 2:
            return mu[-1], logvar[-1]
        else:
            B = mu.shape[1]
            mu, logvar = torch.cat([mu[1,:B//2], mu[2, B//2:]]), \
                         torch.cat([logvar[1,:B//2], logvar[2, B//2:]])
            return mu, logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
