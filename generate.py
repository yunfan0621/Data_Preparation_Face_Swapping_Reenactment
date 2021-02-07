import os
import argparse
import numpy as np

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

from pdb import set_trace as ST

def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            output_dict = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)

            # save the image
            utils.save_image(output_dict['image'], 
                             os.path.join(args.data_dir, 'image_{}'.format(args.size), f'{str(i).zfill(6)}'+'.png'), 
                             nrow=1, normalize=True, range=(-1, 1))

            # save other state tensors
            w_latent = g_ema.get_latent(sample_z).detach()
            state = {
                'z': sample_z,
                'w': w_latent,
                'w_plus': w_plus,
                'c': core_tensor
            }
            torch.save(state, os.path.join(args.data_dir, 'latent_{}'.format(args.size), f'{str(i).zfill(6)}'+'.pt'))
            
if __name__ == '__main__':
    device = 'cuda'

    img_size = 256
    if img_size == 256:
        ckpt_path = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/550000.pt'
    elif img_size == 1024:
        ckpt_path = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/stylegan2-ffhq-config-f.pt'
    else:
        raise ValueError("Invalid image size!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=img_size)
    parser.add_argument('--sample', type=int, default=2)
    parser.add_argument('--pics', type=int, default=1)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=10000)
    parser.add_argument('--ckpt_path', type=str, default=ckpt_path)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='/data2/yunfan.liu/Style_Intervention/sample/')

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    # load the network
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.ckpt_path) # 'g_ema', 'latent_avg'
    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)

    # compute the mean of latent for truncation
    mean_latent_path = os.path.join(args.data_dir, 'latent_code_mean_{}.pt'.format(args.size))
    if args.truncation < 1:
        with torch.no_grad():
            if os.path.exists(mean_latent_path):
                mean_latent = torch.load(mean_latent_path)
            else:
                mean_latent = g_ema.mean_latent(args.truncation_mean)
                torch.save(mean_latent, mean_latent_path)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)