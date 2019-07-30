from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset

from datasets import CUBImageFt, CUBSentences


maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocab_path = '../data/cub/oc:{}_sl:{}_s:{}_w:{}/cub.vocab'.format(minOccur, maxSentLen, 300, lenWindow)
if os.path.exists(vocab_path):
    with open(vocab_path, 'r') as vocab_file:
        vocab = json.load(vocab_file)
    vocabSize = len(vocab['i2w'])  # birds vocabulary
else:
    vocabSize = 1590

eta = 1e-6


def resampler(dataset, idx):
    return idx // 10

class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, expert):
        super(MVAE, self).__init__()
        self.image_enc = ImageEncoder(n_latents)
        self.image_dec = ImageDecoder(n_latents)
        self.sent_enc  = SentEncoder(n_latents)
        self.sent_dec  = SentDecoder(n_latents)
        self.experts       = MixtureOfExperts() if expert=="moe" else ProductOfExperts()
        self.n_latents     = n_latents


    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        # load base datasets
        t1, s1 = self.image_enc.getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.sent_enc.getDataLoaders(batch_size, shuffle, device)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = DataLoader(TensorDataset([
            ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
            t2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(TensorDataset([
            ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
            s2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
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
        img_recon  = self.image_dec(z)
        txt_recon  = self.sent_dec(z)
        return img_recon, txt_recon, mu, logvar

    def infer(self, image=None, sent=None):
        batch_size = image.size(0) if image is not None else sent.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)
        if image is not None:
            img_mu, img_logvar = self.image_enc(image)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if sent is not None:
            txt_mu, txt_logvar = self.sent_enc(sent)
            mu     = torch.cat((mu, txt_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, txt_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

class ImageEncoder(nn.Module):

    """ Generate latent parameters for MNIST image data. """
    def __init__(self, latent_dim, n_c=2048):
        super(ImageEncoder, self).__init__()
        dim_hidden = 256
        self.enc = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            self.enc.add_module("layer" + str(i), nn.Sequential(
                nn.Linear(n_c // (2 ** i), n_c // (2 ** (i + 1))),
                nn.ELU(inplace=True),
            ))
        # relies on above terminating at dim_hidden
        self.fc21 = nn.Linear(dim_hidden, latent_dim)
        self.fc22 = nn.Linear(dim_hidden, latent_dim)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

        train_loader = torch.utils.data.DataLoader(
            CUBImageFt('data', 'train', device),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            CUBImageFt('data', 'test', device),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def forward(self, x):
        e = self.enc(x)
        mu, logvar = self.fc21(e), self.fc22(e)
        return mu, F.softplus(logvar) + eta

class ImageDecoder(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """
    def __init__(self, latent_dim, n_c=2048):
        super(ImageDecoder, self).__init__()
        self.n_c = n_c
        dim_hidden = 256
        self.dec = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            indim = latent_dim if i == 0 else dim_hidden * i
            outdim = dim_hidden if i == 0 else dim_hidden * (2 * i)
            self.dec.add_module("out_t" if i == 0 else "layer" + str(i) + "_t", nn.Sequential(
                nn.Linear(indim, outdim),
                nn.ELU(inplace=True),
            ))
        # relies on above terminating at n_c // 2
        self.fc31 = nn.Linear(n_c // 2, n_c)
        self.fc32 = nn.Linear(n_c // 2, n_c)

    def forward(self, z):
        p = self.dec(z.view(-1, z.size(-1)))
        mean = self.fc31(p).view(*z.size()[:-1], -1)
        return mean


class SentEncoder(nn.Module):
    """ Generate latent parameters for SVHN image data. """
    def __init__(self, latentDim):
        super(SentEncoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.enc = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase * 4, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 8, fBase * 16, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 16, latentDim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 16, latentDim, 4, 1, 0, bias=True)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = lambda data: torch.Tensor(data)
        t_data = CUBSentences('data', split='train', transform=tx, max_sequence_length=32)
        s_data = CUBSentences('data', split='test', transform=tx, max_sequence_length=32)

        train_loader = DataLoader(t_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(s_data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def forward(self, x):
        e = self.enc(self.embedding(x.long()).unsqueeze(1))
        mu, logvar = self.c1(e).squeeze(), self.c2(e).squeeze()
        return mu, F.softplus(logvar) + eta


class SentDecoder(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """
    def __init__(self, latent_dim):
        super(SentDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=True),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:])).view(-1, embeddingDim)

        return self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize)


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
