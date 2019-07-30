from model import MVAE

import torch
import sys
import shutil
from datasets import CUBImageFt
from torchvision import transforms, models, datasets

import matplotlib.pyplot as plt
import os

import numpy as np
fn_2i = lambda t: t.cpu().numpy().astype(int)
fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s


import json

maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
lenWindow = 3
vocab_path = 'data/oc:{}_sl:{}_s:{}_w:{}/cub.vocab'.format(minOccur, maxSentLen, 300, lenWindow)

with open(vocab_path, 'r') as vocab_file:
    vocab = json.load(vocab_file)
i2w = vocab['i2w']


def elbo_loss(recon_image, image, recon_text, text, mu, logvar,
              lambda_mnist=1.0, lambda_svhn=1.0, annealing_factor=1):
    """Bimodal ELBO loss function.

    @param recon_image: torch.Tensor
                        reconstructed image
    @param image: torch.Tensor
                  input image
    @param recon_text: torch.Tensor
                       reconstructed text probabilities
    @param text: torch.Tensor
                 input text (one-hot)
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    @param lambda_mnist: float [default: 1.0]
                         weight for mnist BCE
    @param lambda_svhn: float [default: 1.0]
                       weight for svhn BCE
    @param annealing_factor: integer [default: 1]
                             multiplier for KL divergence term
    @return ELBO: torch.Tensor
                  evidence lower bound
    """
    image_bce, text_bce = 0, 0  # default params
    if recon_image is not None and image is not None:
        image_bce = l2_log_prob(recon_image.view(-1, 2048), image.view(-1, 2048))

    if recon_text is not None and text is not None:
        text_bce = torch.sum(categorical_log_prob(recon_text, text))

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_mnist * image_bce + lambda_svhn * text_bce
                      + annealing_factor * KLD)
    return ELBO


def l2_log_prob(input, target):
    return F.mse_loss(input, target, reduction='sum')

import torch.nn.functional as F

def categorical_log_prob(input, target):
    lpx_z = F.cross_entropy(input=input.view(-1, input.size(-1)),
                             target=target.expand(input.size()[:-1]).long().view(-1),
                             reduction='none',
                             ignore_index=0)

    return lpx_z.view(*input.shape[:-1])

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def unpack_data(dataB, device='cuda'):
    # todo: this can be cleaner...
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)
            elif dataB[0].shape == dataB[1].shape:  # BBSKftPft
                return dataB
            else:  # mnist, svhn, cubI
                return dataB[0].to(device)

        elif is_multidata(dataB[0]) and is_multidata(dataB[1]):  # mnist-svhn, cubIS
            return [d.to(device) for d in list(zip(*dataB))[0]]
        else:
            raise RuntimeError('Unknown type for {}'.format(dataB))
    elif torch.is_tensor(dataB):  # cubS, cubI
        return dataB.to(device)
    else:
        raise RuntimeError('Unknown type for {}'.format(dataB))


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MVAE(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def resize_img(img, ref):
    tx = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(ref.size(-1)),  # use last dim assuming square image
        transforms.ToTensor()
    ])
    return torch.stack([tx(_img) for _img in img.cpu()]).to(img.device).expand_as(ref)

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def unpack_data_mlp(dataB, option='both'):
    if len(dataB[0])==2:
        if option == 'both':
            return dataB[0][0], dataB[1][0], dataB[1][1]
        elif option == 'svhn':
            return dataB[1][0], dataB[1][1]
        elif option == 'mnist':
            return dataB[0][0], dataB[0][1]
    else:
        return dataB


def sent_unproj(sentences):
    """make sure raw data is always passed as dim=2 to avoid argmax.
    last dimension must always be word embedding."""

    if len(sentences.shape) > 2:
        sentences = sentences.argmax(-1).squeeze()
    return [fn_trun(s) for s in fn_2i(sentences)]



def pdist(sample_1, sample_2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n_2, d)``.
    norm : float
        The l_p norm to be used.
    batched : bool
        whether data is batched

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    if len(sample_1.shape) == 2:
        sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
    B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
    norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n_1, n_2)
             + norms_2.transpose(1, 2).expand(B, n_1, n_2))
    distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent

def NN_lookup(emb_h, emb, data):
    indices = pdist(emb.to(emb_h.device), emb_h).argmin(dim=0)
    # indices = torch.tensor(cosine_similarity(emb, emb_h.cpu().numpy()).argmax(0)).to(emb_h.device).squeeze()
    return data[indices]

def projections_dispatch(cuda=True):
    device = torch.device("cuda" if cuda else "cpu")
    train_dataset = CUBImageFt('data', split='train', device=device)
    test_dataset = CUBImageFt('data', split='test', device=device)
    train_dataset._load_data()
    test_dataset._load_data()
    unproj = lambda emb_h, search_split='train', te=train_dataset.ft_mat, td=train_dataset.data_mat, \
                    se=test_dataset.ft_mat, sd=test_dataset.data_mat: \
        NN_lookup(emb_h, te, td) if search_split == 'train' else NN_lookup(emb_h, se, sd)
    return unproj

def imshow(image, caption, i, fig, N):
    """Imshow for Tensor."""
    ax = fig.add_subplot(N // 2, 4, i * 2 + 1)
    ax.axis('off')
    image = image.numpy().transpose((1, 2, 0))  #
    plt.imshow(image)
    ax = fig.add_subplot(N // 2, 4, i * 2 + 2)
    pos = ax.get_position()
    ax.axis('off')
    plt.text(
        x=0.5 * (pos.x0 + pos.x1),
        y=0.5 * (pos.y0 + pos.y1),
        ha='left',
        s='{}'.format(
            ' '.join(i2w[str(i)] + '\n' if (n + 1) % 5 == 0 \
                         else i2w[str(i)] for n, i in enumerate(caption))),
        fontsize=6,
        verticalalignment='center',
        horizontalalignment='center'
    )
    return fig