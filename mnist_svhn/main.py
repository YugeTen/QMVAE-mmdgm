from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import datetime

import numpy as np
from numpy import sqrt
from pathlib import Path
from tempfile import mkdtemp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid

import umap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

from model import MVAE
from utils import elbo_loss, AverageMeter, unpack_data, \
    save_checkpoint,resize_img, Logger, unpack_data_mlp, load_checkpoint

class LatentMLP(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, in_n, out_n):
        super(LatentMLP, self).__init__()
        self.mlp = nn.Linear(in_n, out_n)
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_n, 10),
        #     nn.ReLU(True),
        #     # nn.Linear(100, 100),
        #     # nn.ReLU(True),
        #     nn.Linear(10, 10)
        # )

    def forward(self, x):
        return self.mlp(x)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=20,
                        help='size of the latent embedding [default: 20]')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 128]')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lambda-mnist', type=float, default=3.92,
                        help='scaling for mnist reconstruction [default: 3.92]')
    parser.add_argument('--lambda-svhn', type=float, default=1.,
                        help='scaling for svhn reconstruction [default: 1]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    os.makedirs('./trained_models', exist_ok=True)
    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    sys.stdout = Logger('{}/run.log'.format(runPath))
    print('Expt:', runPath)
    print('RunID:', runId)

    model     = MVAE(args.n_latents)
    model.to(device)
    train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)
    N_mini_batches = len(train_loader)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train(epoch):
        model.train()
        train_loss_meter = AverageMeter()

        # NOTE: is_paired is 1 if the example is paired
        for batch_idx, dataT in enumerate(train_loader):
            mnist, svhn = unpack_data(dataT, device=device)

            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            batch_size = len(mnist)
            optimizer.zero_grad()
            recon_mnist_1, recon_svhn_1, mu_1, logvar_1 = model(mnist, svhn)
            recon_mnist_2, recon_svhn_2, mu_2, logvar_2 = model(mnist)
            recon_mnist_3, recon_svhn_3, mu_3, logvar_3 = model(text=svhn)

            # compute ELBO for each data combo
            joint_loss = elbo_loss(recon_mnist_1, mnist, recon_svhn_1, svhn, mu_1, logvar_1,
                                   lambda_mnist=args.lambda_mnist, lambda_svhn=args.lambda_svhn,
                                   annealing_factor=annealing_factor)
            mnist_loss = elbo_loss(recon_mnist_2, mnist, None, None, mu_2, logvar_2,
                                   lambda_mnist=args.lambda_mnist, lambda_svhn=args.lambda_svhn,
                                   annealing_factor=annealing_factor)
            svhn_loss  = elbo_loss(None, None, recon_svhn_3, svhn, mu_3, logvar_3,
                                   lambda_mnist=args.lambda_mnist, lambda_svhn=args.lambda_svhn,
                                   annealing_factor=annealing_factor)
            train_loss = joint_loss + mnist_loss + svhn_loss
            train_loss_meter.update(train_loss.data.item(), batch_size)
            
            # compute gradients and take step
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_idx * len(mnist), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


    def test():
        model.eval()
        test_loss_meter = AverageMeter()

        for batch_idx, dataT in enumerate(test_loader):
            mnist, svhn = unpack_data(dataT, device=device)

            batch_size = len(mnist)

            with torch.no_grad():
                recon_mnist_1, recon_svhn_1, mu_1, logvar_1 = model(mnist, svhn)
                recon_mnist_2, recon_svhn_2, mu_2, logvar_2 = model(mnist)
                recon_mnist_3, recon_svhn_3, mu_3, logvar_3 = model(text=svhn)

                joint_loss = elbo_loss(recon_mnist_1, mnist, recon_svhn_1, svhn, mu_1, logvar_1)
                mnist_loss = elbo_loss(recon_mnist_2, mnist, None, None, mu_2, logvar_2)
                svhn_loss  = elbo_loss(None, None, recon_svhn_3, svhn, mu_3, logvar_3)
                test_loss  = joint_loss + mnist_loss + svhn_loss
                test_loss_meter.update(test_loss.item(), batch_size)

        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        return test_loss_meter.avg


    def generate(N, K):
        zss, gt = _sample(N)
        gt[0] = resize_img(gt[0], gt[1])

        for key, (mu, std) in zss.items():
            # sample from particular gaussian by multiplying + adding
            if key == 'gen_samples':
                sample = torch.randn(N * N, model.n_latents).to(device)
                sample = sample.mul(std).add_(mu)
            else:
                sample = torch.randn(N, model.n_latents).to(device)
                sample = sample.mul(std).add_(mu)

            # generate
            mnist_mean = torch.sigmoid(model.mnist_dec(sample)).view(-1, 1, 28, 28)
            mnist_std = torch.tensor([0.1]).expand_as(mnist_mean).to(device)
            svhn_mean = torch.sigmoid(model.svhn_dec(sample)).view(-1, 3, 32, 32)
            svhn_std = torch.tensor([0.1]).expand_as(svhn_mean).to(device)

            if key == 'gen_samples':
                mnist_sample = torch.randn((K,*(mnist_mean.size()))).to(device)
                mnist_gen = mnist_sample.mul(mnist_std).add_(mnist_mean).transpose(0,1)
                ms = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in mnist_gen.data.cpu()]
                save_image(torch.stack(ms), '{}/gen_samples_0_{:03d}.png'.format(runPath, epoch), nrow=N)

                svhn_sample = torch.randn((K,*(svhn_mean.size()))).to(device)
                svhn_gen = svhn_sample.mul(svhn_std).add_(svhn_mean).transpose(0,1)
                ss = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in svhn_gen.data.cpu()]
                save_image(torch.stack(ss), '{}/gen_samples_1_{:03d}.png'.format(runPath, epoch), nrow=N)
            else:
                gt_idx = key.split('_')[-1]
                mnist_recon = torch.cat([gt[int(gt_idx)].cpu(), resize_img(mnist_mean.cpu().data, svhn_mean)])
                svhn_recon = torch.cat([gt[int(gt_idx)].cpu(), svhn_mean.cpu().data])
                save_image(mnist_recon, '{}/{}x0_{:03d}.png'.format(runPath, key, epoch))
                save_image(svhn_recon, '{}/{}x1_{:03d}.png'.format(runPath, key, epoch))

    def analyse(N, K):
        z_samples_dict, _ = _sample(N)
        z_samples_dict['gen_samples'] = [torch.zeros((N, model.n_latents)).to(device),
                                         torch.ones((N, model.n_latents)).to(device)]
        z_samples = []

        # draw K samples from each latent
        for key, z in z_samples_dict.items():
            sample = torch.randn(K, N, model.n_latents).to(device)
            mu, std = z
            z_samples.append(sample.mul(std).add_(mu).view(-1, model.n_latents))
        z_labels = [torch.zeros_like(zs[:, 0]) + i for i, zs in enumerate(z_samples)]
        zss = torch.cat(tuple(z_samples), dim=0).cpu().detach().numpy()
        zls = torch.cat(tuple(z_labels), dim=0).cpu().detach().numpy()
        z_legends = ['Prior', 'MNIST', 'SVHN', 'MNIST + SVHN']

        embedding = umap.UMAP(n_neighbors=5,
                              min_dist=0.1,
                              metric='correlation').fit_transform(zss)
        cmap_array = np.concatenate((plt.cm.Set2([7]),plt.cm.Set1([2,1,0])), axis=0)
        color_map = colors.LinearSegmentedColormap.from_list('new_cmap', cmap_array)
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=zls, cmap=color_map, s=5)
        legend_elements = []
        for i, (legend, cm) in enumerate(zip(z_legends, cmap_array)):
            legend_elements.append(Line2D([0], [0], marker='o', color=cm, label=legend))
        plt.legend(handles=legend_elements)
        plt.savefig('{}/latents_{:03d}.png'.format(runPath, epoch))

    def _sample(N):
        model.eval()
        for batch_idx, dataT in enumerate(test_loader):
            mnist, svhn = unpack_data(dataT, device=device)
            break
        gt = [mnist[:N], svhn[:N], torch.cat([resize_img(mnist[:N], svhn[:N]), svhn[:N]])]
        zss = OrderedDict()

        # mode 1: generate
        zss['gen_samples'] = [torch.zeros((N * N, model.n_latents)).to(device),
                              torch.ones((N * N, model.n_latents)).to(device)]

        # mode 2: mnist --> mnist, mnist --> svhn
        mu, logvar = model.infer(mnist=gt[0])
        zss['recon_0'] = [mu, logvar.mul(0.5).exp_()]

        # mode 3: svhn --> mnist, svhn --> svhn
        mu, logvar = model.infer(svhn=gt[1])
        zss['recon_1'] = [mu, logvar.mul(0.5).exp_()]

        # mode 4: mnist, svhn --> mnist, mnist, svhn --> svhn
        mu, logvar = model.infer(mnist=gt[0], svhn=gt[1])
        zss['recon_2'] = [mu, logvar.mul(0.5).exp_()]
        return zss, gt

    def latent_classification(epochs):
        model=load_checkpoint('trained_models/model_best.pth.tar')
        model.eval()

        classifier = LatentMLP(20, 10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            total_iters = len(train_loader)
            print('\n====> Epoch: {:03d} '.format(epoch))
            for i, data in enumerate(train_loader):
                # get the inputs
                svhn, targets = unpack_data_mlp(data, option='svhn')
                svhn, targets = svhn.to(device), targets.to(device)
                # mnist, svhn, targets = unpack_data_mlp(data, option='both')
                # mnist, svhn, targets = mnist.to(device), svhn.to(device), targets.to(device)
                with torch.no_grad():
                    # mu, logvar = model.infer(mnist=mnist, svhn=svhn)
                    mu, logvar = model.infer(svhn=svhn)
                    zss = [mu, logvar.mul_(0.5).exp_()]
                sample = torch.randn(mu.shape).to(device)
                sample.mul_(zss[1]).add_(zss[0]).view(-1, mu.shape[-1])
                optimizer.zero_grad()
                outputs = classifier(sample)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (i + 1) % 1000 == 0:
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == targets).sum().item()
                    print('iteration {:04d}/{:d}: loss: {:6.3f},'
                          ' acc: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000, correct/targets.size(0)))
                    running_loss = 0.0
        print('Finished Training, calculating test loss...')

        classifier.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # mnist, svhn, targets = unpack_data_mlp(data, option='both')
                # mnist, svhn, targets = mnist.to(device), svhn.to(device), targets.to(device)
                # mu, logvar = model.infer(mnist=mnist, svhn=svhn)
                svhn, targets = unpack_data_mlp(data, option='svhn')
                svhn, targets = svhn.to(device), targets.to(device)
                mu, logvar = model.infer(svhn=svhn)
                zss = [mu, logvar.mul_(0.5).exp_()]
                sample = torch.randn(mu.shape).to(device)
                sample.mul_(zss[1]).add_(zss[0]).view(-1, mu.shape[-1])
                outputs = classifier(sample)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print('The classifier correctly classified {} out of {} examples. Accuracy: '
              '{:.2f}%'.format(correct, total, (correct / total) * 100))


    best_loss = sys.maxsize
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        generate(N=8, K=9) # N: zs sample, K: px_z sample
        analyse(N=args.batch_size, K=10)
        test_loss = test()
        is_best   = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=runPath)
    # latent_classification(1)