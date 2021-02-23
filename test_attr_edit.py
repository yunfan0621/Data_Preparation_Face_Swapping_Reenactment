import os
import argparse
import numpy as np
import base64

import torch
from torchvision import utils

from model import Generator
from utils import latent_interpolate, check_mkdir
from PythonSDK.facepp import API, File

from pdb import set_trace as ST

api = API()


def convert_image(img):
	img = img.clone().cpu().numpy()


def id_check(img1, img2, th=76.5):
	global api

	img_str = cv2.imencode('.jpg', img_rgb_resize)[1].tostring()
	img_base64 = base64.b64encode(img_str)


def check_disentanglement(img_list, attr_name):
	pass


if __name__ == '__main__':
	steps = 5
	img_size = 256
	data_root = os.path.join('/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/synthetic_dataset/{}'.format(img_size))
	svm_root  = os.path.join('/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset/svm_normals_{}'.format(img_size))

	attr_name_list   = ['smile', 'yaw', 'happiness', 'mouth_open'] # 'pitch', 'roll', 'eyes_open',
	space_name_list  = ['style', 'w',   'style',     'style']      # 'w', 'w', 'style',
	regularizer_list = ['l1',    'l2',  'l1',        'l1']         # 'l2', 'l2', 'l1',
	end_point_list   = [10.0,    5.0,   10.0,        10.0]         # 3.0, 10.0, 50.0,

	input_latent_dir = os.path.join(data_root, 'origin_latent')
	input_img_dir    = os.path.join(data_root, 'origin_image')
	# edit_latent_dir  = os.path.join(data_root, 'edit_latent')
	# edit_img_dir     = os.path.join(data_root, 'edit_image')
	edit_img_dir     = os.path.join(data_root, 'test_synthesis')

	# load the generator
	if img_size == 1024:
		ckpt_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/stylegan2-ffhq-config-f.pt"
	elif img_size == 256:
		ckpt_path = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/checkpoint/550000.pt"
	else:
		raise ValueError('Invalid image size!')

	g_ema = Generator(img_size, 512, 8, channel_multiplier=2).to('cuda')
	checkpoint = torch.load(ckpt_path)
	g_ema.load_state_dict(checkpoint['g_ema'], strict=False)
	with torch.no_grad():
		mean_latent = g_ema.mean_latent(10000)

	latent_list = os.listdir(input_latent_dir)
	for attr_name, space_name, regularizer, end_point in zip(attr_name_list, space_name_list, regularizer_list, end_point_list):
		
		# skip for tested attribute
		if attr_name == 'smile' or attr_name == 'yaw' or attr_name == 'happiness':
			continue

		svm_path = os.path.join(svm_root, space_name, regularizer, 'n_{}.pt'.format(attr_name))
		svm_data = torch.load(svm_path)
		w = svm_data['w']
		b = svm_data['b']

		# create the saving directory
		save_img_dir = os.path.join(edit_img_dir, attr_name)
		check_mkdir(save_img_dir)

		for latent_name in latent_list:
			latent_path = os.path.join(input_latent_dir, latent_name)
			latent_data = torch.load(latent_path)
			latent_code = latent_data[space_name]
			core_tensor = latent_data['c']

			# generate the edited attribute and latent
			latent_code_list = latent_interpolate(latent_code, w, b, latent_mode=space_name, end_point=end_point, steps=steps)
			img_list = []
			for latent_code in latent_code_list:
				output_dict = g_ema([latent_code], truncation=0.5, truncation_latent=mean_latent, input_mode=space_name, core_tensor=core_tensor)
				img_list.append(output_dict['image'].squeeze(0).detach())

			# save the interpolation result to disk
			# save_img_name = '{}_{}_{}.png'.format(latent_name[:-3], space_name, end_point)
			save_img_name = latent_name[:-3] + '.png'
			save_img_path = os.path.join(save_img_dir, save_img_name)
			utils.save_image(img_list, save_img_path, nrow=len(img_list), normalize=True, range=(-1, 1))