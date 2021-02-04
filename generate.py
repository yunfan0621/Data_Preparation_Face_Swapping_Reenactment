import os
import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            utils.save_image(sample, os.path.join(args.save_path, f'{str(i).zfill(6)}.png'), nrow=1, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=1)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=10000)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    # load the model according to image resolution
    if args.size == 1024:
        args.ckpt_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/stylegan2-ffhq-config-f.pt"
        args.save_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/image_1024"
    elif args.size == 256:
        args.ckpt_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/550000.pt"
        args.save_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/image_256"
    else:
        raise ValueError('Invalid image size!')

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.ckpt_path)
    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)

    # compute the mean of latent space
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)