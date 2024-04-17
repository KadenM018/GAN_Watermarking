import argparse
import copy
import random
import math

import torchvision
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

parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
parser.add_argument('--size', type=int, default=128, help='size of the image')
parser.add_argument('--num_samples', type=int, default=16)
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--decoder_weights', type=str,
                    default=r"C:\Users\kaden\Downloads\stegastamp_100_6000_decoder.pth")

parser.add_argument('--stylegan_weights', type=str,
                    default=r"C:\Users\kaden\Downloads\00076000.pth")

parser.add_argument('--fingerprints', type=str,
                    default=r"C:\Users\kaden\Downloads\fingerprints.pth")

parser.add_argument('--save_path', type=str,
                    default=r"C:\Users\kaden\Main\EE465\GAN_Watermarking\stylegan+decoder\test_results_kaden")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'
code_size = 512
b_size = args.num_samples
step = int(math.log2(args.size)) - 2

generator = nn.DataParallel(StyledGenerator(code_size)).cuda()

ckpt = torch.load(args.stylegan_weights)
generator.module.load_state_dict(ckpt['generator'])

decoder = StegaStampDecoder(resolution=args.size, IMAGE_CHANNELS=3, fingerprint_size=100)
weights = torch.load(args.decoder_weights)
decoder.load_state_dict(weights)
decoder.cuda()
decoder.eval()

fingerprints = torch.load(args.fingerprints).cuda()

gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
                2, 0
            )
gen_in1 = gen_in1.squeeze(0)
gen_in2 = gen_in2.squeeze(0)

fake_image = generator(gen_in2, step=step, alpha=1)

decoder_output = decoder(fake_image)

criterion = nn.BCEWithLogitsLoss()
BCE_loss = 0
for j in range(0, decoder_output.size(0)):
    BCE_loss += criterion(decoder_output[j].view(-1), fingerprints.view(-1))
BCE_loss = BCE_loss / decoder_output.size(0)
print(f'\nBCE_loss: {BCE_loss}\n')

fingerprints_predicted = (decoder_output > 0).float()
bitwise_accuracy = 1.0 - torch.mean(
    torch.abs(fingerprints - fingerprints_predicted)
)
print(f'\nbitwise accuracy: {bitwise_accuracy}\n')

torchvision.utils.save_image(fake_image, os.path.join(args.save_path, 'sample.png'), normalize=True)
