import os
import sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from glob import glob
import time
import datetime
import imageio
from tqdm import tqdm

import argparse

import face_alignment
from skimage import io
from utils import get_parsing_map

from pdb import set_trace as ST

# import FLAME models
sys.path.append('/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/')
from renderer import Renderer
import util

sys.path.append('/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/models/')
from FLAME import FLAME, FLAMETex

torch.backends.cudnn.benchmark = True

config = {
'flame_model_path': '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/data/generic_model.pkl',
'flame_lmk_embedding_path': '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/data/landmark_embedding.npy',
'tex_space_path': '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/data/FLAME_texture.npz',

# FLAME parameters
'camera_params': 3,
'shape_params': 100,
'expression_params': 50,
'pose_params': 6,
'tex_params': 50,
'use_face_contour': True,

# experiment parameters
'cropped_size': 256, # for image reading
'batch_size': 1,
'image_size': 224, # for image renderring
'e_lr': 0.005,
'e_wd': 0.0001,

# weights of losses and reg terms
'w_pho': 8,
'w_lmks': 1,
'w_shape_reg': 1e-4,
'w_expr_reg': 1e-4,
'w_pose_reg': 0,

# other controlling parameters
'print_freq': 100,
'verbose': False,
'save_all_output': False
}

def face_recon_FLAME(params_path, crop_size=256, img_size=224, device='cuda'):
	global config
	config = util.dict2obj(config)

	flame = FLAME(config).to(device)
	flametex = FLAMETex(config).to(device)
	mesh_file = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/data/head_template_mesh.obj'
	render = Renderer(img_size, obj_filename=mesh_file).to(device)

	params = np.load(params_path, allow_pickle=True)
	shape = torch.from_numpy(params.item()['shape']).float().to(device)
	exp   = torch.from_numpy(params.item()['exp']).float().to(device)
	pose  = torch.from_numpy(params.item()['pose']).float().to(device)
	tex   = torch.from_numpy(params.item()['tex']).float().to(device)
	light = torch.from_numpy(params.item()['lit']).float().to(device)
	verts = torch.from_numpy(params.item()['verts']).float().to(device) # trans_vertices

	vertices, _, _ = flame(shape_params=shape, expression_params=exp, pose_params=pose)
	albedos = flametex(tex) / 255.
	img_render = render(vertices, verts, albedos, light)['images']

	return img_render


if __name__ == '__main__':
	img_size = 256
	data_root = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/synthetic_dataset/{}/'.format(img_size)

	img_dir = os.path.join(data_root, 'origin_image')
	img_list = os.listdir(img_dir)
	for img_name in img_list:
		img_path = os.path.join(img_dir, img_name)

		img_prefix = img_name[:-4]
		ldmk_path = os.path.join(data_root, 'ldmk', img_prefix+'_ldmk.npy')
		mask_path = os.path.join(data_root, 'mask', img_prefix+'_mask.npy')
		params_dir = os.path.join(data_root, '3DMM_params', img_prefix)
		params_path = os.path.join(params_dir, img_prefix+'.npy')

		if not os.path.exists(ldmk_path) or not os.path.exists(mask_path) or not os.path.exists(params_dir) or not os.path.exists(params_path):
			print('Material for sample {} not complete!'.format(img_name))
			continue

		# read in the material for 3DMM face reconstruction
		ldmk = np.load(ldmk_path)
		mask = np.load(mask_path)
		params = np.load(params_path, allow_pickle=True)

		img = cv2.resize(cv2.imread(img_path), (img_size, img_size)).astype(np.float32) / 255.
		img = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
		img = torch.from_numpy(img)
		torchvision.utils.save_image([img], 'origin_image.png', nrow=1, normalize=True, range=(0, 1))

		img_render = face_recon_FLAME(params_path, crop_size=256, device='cuda')
		torchvision.utils.save_image([img_render.squeeze()], 'recon_image.png', nrow=1, normalize=True, range=(0, 1))

		ST()