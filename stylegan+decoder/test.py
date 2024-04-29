import argparse
import math
import torchvision
import torch
from torch import nn
from model import StyledGenerator, StegaStampDecoder
import os
from torchvision import transforms
import numpy as np

data_processing = transforms.ColorJitter(0, 0, 0, 0)  # (brightness, contrast, saturation, hue)

parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

parser.add_argument('--size', type=int, default=128,
                    help='Size of the generated image. For CelebA this should be 128 and for flower dataset it should be 256')

parser.add_argument('--num_samples', type=int, default=16,
                    help='Number of samples to generate')

parser.add_argument('--device', type=str, default='cuda:2',
                    help='Write cuda:<device_num> for gpu or cpu for cpu')

parser.add_argument('--decoder_weights', type=str,
                    default="/home/kaden/Desktop/GAN_Watermarking/ArtificialGANFingerprints/results/brightness0.75/checkpoints/stegastamp_100_30000_decoder.pth",
                    help='Path to where the decoder weights are saved')

parser.add_argument('--stylegan_weights', type=str,
                    default="/home/kaden/Desktop/GAN_Watermarking/stylegan+decoder/StyleGAN+decoder_weights/checkpoints/brightness0.75/00018000.pth",
                    help='Path to where the StyleGAN weights are saved')

parser.add_argument('--fingerprints', type=str,
                    default="/home/kaden/Desktop/GAN_Watermarking/stylegan+decoder/StyleGAN+decoder_weights/watermarks/brightness0.75/fingerprints.pth",
                    help='Path to where the watermark/fingerprints is saved')

parser.add_argument('--save_path', type=str,
                    default="/home/kaden/Desktop/GAN_Watermarking (another copy)/stylegan+decoder/sample",
                    help='Path of the folder to where the images will be saved')

args = parser.parse_args()

code_size = 512
step = int(math.log2(args.size)) - 2

# Create generator, load weights into memory, then load weights into the generator
if args.device[0:4] == 'cuda':
    generator = nn.DataParallel(StyledGenerator(code_size), device_ids=[int(args.device[-1])]).to(args.device)
    ckpt = torch.load(args.stylegan_weights, map_location=args.device)
    generator.module.load_state_dict(ckpt['generator'])
else:
    generator = StyledGenerator(code_size).to(args.device)
    ckpt = torch.load(args.stylegan_weights, map_location=args.device)
    generator.load_state_dict(ckpt['generator'])

decoder = StegaStampDecoder(resolution=args.size, IMAGE_CHANNELS=3, fingerprint_size=100)  # Create decoder
weights = torch.load(args.decoder_weights, map_location=args.device)  # load weights into memory
decoder.load_state_dict(weights)  # load weights into decoder model
decoder.to(args.device)  # send decoder to the given device
decoder.eval()  # put decoder in evaluation mode (performs differently in training mode)

fingerprints = torch.load(args.fingerprints).to(args.device)  # Load watermark into memory

# Generate random noise
_, gen_in2 = torch.randn(2, args.num_samples, code_size, device=args.device).chunk(2, 0)
gen_in2 = gen_in2.squeeze(0)

# Give random noise to generator and save output
fake_image = generator(gen_in2, step=step, alpha=0)

# Do image processing on the image from the generator
altered_images = data_processing(fake_image)

# Send altered image to the decoder and save output
decoder_output = decoder(altered_images)

# Calculate binary cross-entropy loss (BCE) between predicted watermark and actual watermark
criterion = nn.BCEWithLogitsLoss()
BCE_loss = 0
for j in range(0, decoder_output.size(0)):
    BCE_loss += criterion(decoder_output[j].view(-1), fingerprints.view(-1))
BCE_loss = BCE_loss / decoder_output.size(0)
print(f'\nBCE_loss: {BCE_loss}')

# Convery predicted watermark to binary numbers and calculate bitwise accuracy
fingerprints_predicted = (decoder_output > 0).float()
bitwise_acc_arr = 1.0 - torch.mean(torch.abs(fingerprints - fingerprints_predicted), dim=1)
bitwise_accuracy = 1.0 - torch.mean(torch.abs(fingerprints - fingerprints_predicted))

torch.save(fingerprints_predicted, '/home/kaden/Desktop/GAN_Watermarking/fingertest.pth')

print(f'\naverage bitwise accuracy: {bitwise_accuracy}')

# Save samples as a batch in one image
torchvision.utils.save_image(fake_image, os.path.join(args.save_path, 'sample.png'), normalize=True)

# Save samples as individual images
# import matplotlib.pyplot as plt
# for i, image in enumerate(fake_image):
#     img = image.permute(1, 2, 0).detach().cpu().numpy()
#     img = (img - np.min(img)) / (np.max(img) - np.min(img))
#     img *= 255
#     img = np.uint8(img)
#     plt.imsave(os.path.join(args.save_path, f'sample_{i}.png'), img)

print('\nActual Watermark:')
print(fingerprints.cpu().numpy())
print('\nPredicted Watermark')
print(fingerprints_predicted[np.argmax(bitwise_acc_arr.cpu().numpy())].cpu().numpy())
