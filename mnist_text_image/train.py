from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.optim as optim
from torchvision.utils import save_image

import os
import sys
import datetime
from pathlib import Path
from tempfile import mkdtemp
from collections import OrderedDict

from model import MVAE
from utils import elbo_loss, AverageMeter, unpack_data, \
    save_checkpoint,resize_img, Logger


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
    parser.add_argument('--lambda-image', type=float, default=3.92,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--lambda-text', type=float, default=1.,
                        help='multipler for text reconstruction [default: 10]')
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
            image, text = unpack_data(dataT, device=device)

            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            batch_size = len(image)

            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through model
            recon_image_1, recon_text_1, mu_1, logvar_1 = model(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = model(text=text)
                
            # compute ELBO for each data combo
            joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1, 
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                   annealing_factor=annealing_factor)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, 
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                   annealing_factor=annealing_factor)
            text_loss  = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3, 
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                  annealing_factor=annealing_factor)
            train_loss = joint_loss + image_loss + text_loss
            train_loss_meter.update(train_loss.data.item(), batch_size)
            
            # compute gradients and take step
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


    def test():
        model.eval()
        test_loss_meter = AverageMeter()

        for batch_idx, dataT in enumerate(test_loader):
            image, text = unpack_data(dataT, device=device)

            batch_size = len(image)

            with torch.no_grad():
                recon_image_1, recon_text_1, mu_1, logvar_1 = model(image, text)
                recon_image_2, recon_text_2, mu_2, logvar_2 = model(image)
                recon_image_3, recon_text_3, mu_3, logvar_3 = model(text=text)

                joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1)
                image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2)
                text_loss  = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3)
                test_loss  = joint_loss + image_loss + text_loss
                test_loss_meter.update(test_loss.item(), batch_size)

        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        return test_loss_meter.avg

    def generate(N):
        model.eval()
        for batch_idx, dataT in enumerate(test_loader):
            image, text = unpack_data(dataT, device=device)
            break
        gt = [image[:N], text[:N], torch.cat([resize_img(image[:N], text[:N]), text[:N]])]
        zss = OrderedDict()

        # mode 1: generate
        zss['gen_samples'] = [torch.zeros((N * N, model.n_latents)).to(device),
                              torch.ones((N * N, model.n_latents)).to(device)]

        # mode 2: mnist --> mnist, mnist --> svhn
        mu, logvar = model.infer(image=gt[0])
        zss['recon_0'] = [mu, logvar.mul(0.5).exp_()]

        # mode 3: svhn --> mnist, svhn --> svhn
        mu, logvar = model.infer(text=gt[1])
        zss['recon_1'] = [mu, logvar.mul(0.5).exp_()]

        # mode 4: mnist, svhn --> mnist, mnist, svhn --> svhn
        mu, logvar = model.infer(image=gt[0], text=gt[1])
        zss['recon_2'] = [mu, logvar.mul(0.5).exp_()]

        gt[0] = resize_img(gt[0], gt[1])

        for key, (mu, std) in zss.items():
            # sample from particular gaussian by multiplying + adding
            if key == 'gen_samples':
                sample = torch.randn(N * N, model.n_latents).to(device)
                sample = sample.mul(std).add_(mu)
            else:
                sample = torch.randn(N, model.n_latents).to(device)
                sample = sample.mul(std).add_(mu)

            # generate image and text
            img_recon = torch.sigmoid(model.mnist_dec(sample)).view(-1, 1, 28, 28).cpu().data
            txt_recon = torch.sigmoid(model.svhn_dec(sample)).view(-1, 3, 32, 32).cpu().data
            img_recon = resize_img(img_recon, txt_recon)

            if key == 'gen_samples':
                save_image(img_recon, '{}/gen_samples_0_{:03d}.png'.format(runPath, epoch), nrow=N)
                save_image(txt_recon, '{}/gen_samples_1_{:03d}.png'.format(runPath, epoch), nrow=N)
            else:

                gt_idx = key.split('_')[-1]
                comp_img = torch.cat([gt[int(gt_idx)].cpu(), img_recon])
                comp_txt = torch.cat([gt[int(gt_idx)].cpu(), txt_recon])
                save_image(comp_img, '{}/{}x0_{:03d}.png'.format(runPath, key, epoch))
                save_image(comp_txt, '{}/{}x1_{:03d}.png'.format(runPath, key, epoch))

    best_loss = sys.maxsize
    for epoch in range(1, args.epochs + 1):
        # train(epoch)
        generate(N=8)
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
