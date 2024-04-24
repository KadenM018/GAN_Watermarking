import argparse
import math
import torchvision
import torch
from torch import nn
from model import StyledGenerator,StegaStampDecoder
import os
from torchvision import transforms

parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
parser.add_argument('--size', type=int, default=128,
                    help='Size of the generated image. For CelebA this should be 128')
parser.add_argument('--num_samples', type=int, default=16,
                    help='Number of samples to generate')
parser.add_argument('--device', type=str, default='cpu',
                    help='Write cuda:<device_num> for gpu or cpu for cpu')

parser.add_argument('--decoder_weights', type=str,
                    default="/home/kaden/Downloads/stegastamp_100_6000_decoder.pth",
                    help='Path to where the decoder weights are saved')

parser.add_argument('--stylegan_weights', type=str,
                    default="/home/kaden/Desktop/GAN_Watermarking/stylegan+decoder/checkpoints/good_optim1/00076000.pth",
                    help='Path to where the StyleGAN weights are saved')

parser.add_argument('--fingerprints', type=str,
                    default="/home/kaden/Desktop/GAN_Watermarking/stylegan+decoder/fingerprints/good_optim1/fingerprints.pth",
                    help='Path to where the watermark is saved')

parser.add_argument('--save_path', type=str,
                    default="/home/kaden/Desktop/GAN_Watermarking/stylegan+decoder/samples/normal",
                    help='Path of the folder to where the images will be saved')

args = parser.parse_args()

code_size = 512
b_size = args.num_samples
step = int(math.log2(args.size)) - 2

if args.device[0:4] == 'cuda':
    generator = nn.DataParallel(StyledGenerator(code_size), device_ids=[int(args.device[-1])]).to(args.device)
    ckpt = torch.load(args.stylegan_weights, map_location=args.device)
    generator.module.load_state_dict(ckpt['generator'])
else:
    generator = StyledGenerator(code_size).to(args.device)
    ckpt = torch.load(args.stylegan_weights, map_location=args.device)
    generator.load_state_dict(ckpt['generator'])

decoder = StegaStampDecoder(resolution=args.size, IMAGE_CHANNELS=3, fingerprint_size=100)
weights = torch.load(args.decoder_weights, map_location=args.device)
decoder.load_state_dict(weights)
decoder.to(args.device)
decoder.eval()

fingerprints = torch.load(args.fingerprints).to(args.device)

gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device=args.device).chunk(
                2, 0
            )
gen_in1 = gen_in1.squeeze(0)
gen_in2 = gen_in2.squeeze(0)

fake_image = generator(gen_in2, step=step, alpha=1)

# Added stuff
data_processing = transforms.ColorJitter(0, 0, 0, 0)
altered_images = data_processing(fake_image)

decoder_output = decoder(altered_images)

criterion = nn.BCEWithLogitsLoss()
BCE_loss = 0
for j in range(0, decoder_output.size(0)):
    BCE_loss += criterion(decoder_output[j].view(-1), fingerprints.view(-1))
BCE_loss = BCE_loss / decoder_output.size(0)
print(f'\nBCE_loss: {BCE_loss}')

fingerprints_predicted = (decoder_output > 0).float()
bitwise_accuracy = 1.0 - torch.mean(
    torch.abs(fingerprints - fingerprints_predicted)
)
print(f'\nbitwise accuracy: {bitwise_accuracy}')

torchvision.utils.save_image(fake_image, os.path.join(args.save_path, 'sample.png'), normalize=True)
