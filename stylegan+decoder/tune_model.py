import argparse
import copy
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator, StegaStampDecoder

import os


def generate_random_fingerprints(fingerprint_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    return z

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    dataset.transform.transforms[2] = transforms.Resize(dataset.resolution)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, dataset, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    pbar = tqdm(range(3_000_000))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    # Create Stegastamp decoder
    decoder = StegaStampDecoder(resolution=args.init_size, IMAGE_CHANNELS=3, fingerprint_size=100)

    # Load weights into model and freeze weights
    weights = torch.load(args.segastamp_weights)
    decoder.load_state_dict(weights)
    decoder.cuda()
    decoder.eval()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./fingerprints', exist_ok=True)

    # Create Fingerprints
    fingerprints = generate_random_fingerprints(100, 1, (args.init_size, args.init_size))
    # fingerprints = copy.deepcopy(fingerprint_base)
    torch.save(fingerprints, './fingerprints/fingerprints.pth')
    # for i in range(0, args.batch_size - 1):
    #     fingerprints = torch.concatenate([fingerprints, fingerprint_base])
    fingerprints = fingerprints.cuda()

    for i in pbar:

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        b_size = real_image.size(0)
        real_image = real_image.cuda()

        gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
            2, 0
        )
        gen_in1 = gen_in1.squeeze(0)
        gen_in2 = gen_in2.squeeze(0)

        fake_image_1 = generator(gen_in1, step=step, alpha=alpha)

        # Added stuff
        decoder_output = decoder(fake_image_1)
        criterion = nn.BCEWithLogitsLoss()

        BCE_loss = 0
        for j in range(0, decoder_output.size(0)):
            BCE_loss += criterion(decoder_output[j].view(-1), fingerprints.view(-1))
        BCE_loss = BCE_loss / decoder_output.size(0)

        if (i + 1) % n_critic == 0:

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            # fake_image_2 = generator(gen_in2, step=step, alpha=alpha)

            loss = BCE_loss
            print(loss.item())

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            if (i + 1) % 25 == 0:
                fingerprints_predicted = (decoder_output > 0).float()
                bitwise_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - fingerprints_predicted)
                )
                print(f'bitwise accuracy: {bitwise_accuracy}\n')

        if (i + 1) % 250 == 0:  # 100
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            torch.randn(gen_j, code_size).cuda(), step=step, alpha=alpha
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(images, 0),
                f'./samples/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('--path', type=str, help='path of specified dataset',
                        default=r"C:\Users\kaden\Main\EE465\CelebA\img_align_celeba\img_align_celeba")
    parser.add_argument('--segastamp_weights', type=str, help='Path to trained StegaStamp weights',
                        default=r"C:\Users\kaden\Main\EE465\Watermarking_Project1\resutls\test_stegastamp_AGANF\checkpoints\stegastamp_100_6000_decoder.pth")
    parser.add_argument('--stylegan_weights', type=str, help='Path to trained StyleGAN weights',
                        default='')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--phase', type=int, default=9999999999,
                        help='number of samples used for each training phases')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=128, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--no_from_rgb_activate', action='store_true',
                        help='use activate in from_rgb (original implementation)')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--loss', type=str, default='wgan-gp', choices=['wgan-gp', 'r1'],
                        help='class of gan loss')
    parser.add_argument('--stylegan_ctrl', type=int, default=1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda_device}'

    batch_size = args.batch_size

    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    if os.path.isfile(args.stylegan_weights):
        weights = torch.load(args.stylegan_weights, map_location=f'cuda:0')
        generator.load_state_dict(weights['generator'])
        discriminator.load_state_dict(weights['discriminator'])
        g_running.load_state_dict(weights['g_running'])

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(int(args.init_size * 1.391)),
            transforms.Resize(args.init_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003, 2048: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = args.batch_size

    train(args, dataset, generator, discriminator)
