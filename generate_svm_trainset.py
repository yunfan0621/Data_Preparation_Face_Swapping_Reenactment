import os
import argparse

import torch
from torchvision import utils
from tqdm import tqdm

from model import Generator
from utils import check_mkdir

from pdb import set_trace as ST


def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.n_samples)):
            # set save paths
            save_img_dir = os.path.join(args.save_dir, 'image_{}'.format(args.size))
            check_mkdir(save_img_dir)
            save_img_path = os.path.join(save_img_dir, f'{str(i).zfill(6)}'+'.png')

            save_latent_dir = os.path.join(args.save_dir, 'latent_{}'.format(args.size))
            check_mkdir(save_latent_dir)
            save_latent_path = os.path.join(save_latent_dir, f'{str(i).zfill(6)}'+'.pt')

            if os.path.exists(save_img_path) and os.path.exists(save_latent_path):
                print('Skipping for sample No. {}'.format(i))
                continue

            # create and save
            sample_z = torch.randn(args.sample, args.latent, device=device)
            output_dict = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)
            utils.save_image(output_dict['image'], save_img_path, nrow=1, normalize=True, range=(-1, 1))

            state = {
                'z': sample_z,
                'w': g_ema.get_latent(sample_z),
                'w_plus': output_dict['style_codes'],
                'c': output_dict['core_tensor']
            } # all variables do not require gradient            
            torch.save(state, save_latent_path)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=10000)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    # load the model according to image resolution
    args.save_dir = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset"
    if args.size == 1024:
        args.ckpt_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/stylegan2-ffhq-config-f.pt"
    elif args.size == 256:
        args.ckpt_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/550000.pt"
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