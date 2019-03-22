from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import torch.optim as optim
from torch.autograd import Variable

from utils import *



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


    def test(epoch):
        model.eval()
        test_loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(test_loader):
            if args.cuda:
                image  = image.cuda()
                text   = text.cuda()

            image = Variable(image, volatile=True)
            text  = Variable(text, volatile=True)
            batch_size = len(image)

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

    
    best_loss = sys.maxsize
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test(epoch)
        is_best   = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')   
