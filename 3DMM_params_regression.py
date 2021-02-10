import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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

class PhotometricFitting(object):
    def __init__(self, config, device='cuda'):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.config = config
        self.device = device
        
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)

        self._setup_renderer()

    def _setup_renderer(self):
        mesh_file = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)

    def optimize(self, images, landmarks, image_masks, savefolder=None):
        bz = images.shape[0]
        pose   = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        exp    = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        shape  = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        tex    = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        cam    = torch.zeros(bz, self.config.camera_params); cam[:, 0] = 5.
        cam    = nn.Parameter(cam.float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        
        e_opt  = torch.optim.Adam(
            [shape, exp, pose, cam, tex, lights],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        e_opt_rigid = torch.optim.Adam(
            [pose, cam],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        gt_landmark = landmarks

        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in range(200):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            if self.config.verbose:
                loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                for key in losses.keys():
                    loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

                if k % self.config.print_freq == 0:
                    print(loss_info)

            if self.config.save_all_output and k % self.config.print_freq == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d']  = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                grids['landmarks3d']  = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))

                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                savefolder_iter = os.path.join(savefolder, 'inter_steps') # use a separate folder to save intermediate results
                util.check_mkdir(savefolder_iter)
                cv2.imwrite('{}/{}.jpg'.format(savefolder_iter, k), grid_image)

        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        for k in range(200, 1500):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * config.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * config.w_pose_reg

            ## render
            albedos = self.flametex(tex) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * config.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss

            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            if self.config.verbose:
                loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                for key in losses.keys():
                    loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

                if k % self.config.print_freq == 0:
                    print(loss_info)

            # visualize
            if self.config.save_all_output and k % self.config.print_freq == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d']  = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                grids['landmarks3d']  = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                grids['albedoimage']  = torchvision.utils.make_grid((ops['albedo_images'])[visind].detach().cpu())
                grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                shape_images = self.render.render_shape(vertices, trans_vertices, images)
                grids['shape'] = torchvision.utils.make_grid(F.interpolate(shape_images[visind], [224, 224])).detach().float().cpu()

                # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                savefolder_iter = os.path.join(savefolder, 'inter_steps') # use a separate folder to save intermediate results
                util.check_mkdir(savefolder_iter)
                cv2.imwrite('{}/{}.jpg'.format(savefolder_iter, k), grid_image)

        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'verts': trans_vertices.detach().cpu().numpy(),
            'albedos':albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy()
        }
        return single_params

    def run(self, image_path, landmark_path, mask_path):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        # Photometric optimization is sensitive to the hair or glass occlusions,
        # therefore we use a face segmentation network to mask the skin region out.
        images = []
        landmarks = []
        image_masks = []

        # check and make the save dir, skip if exists
        image_name = os.path.basename(image_path)[:-4]
        savefolder = os.path.sep.join([self.config.save_dir, image_name])
        util.check_mkdir(savefolder)
        if os.path.exists(os.path.join(savefolder, image_name+'.npy')):
        	print('Skipping for {}...'.format(image_name))
        	return

        # read and process the image
        image = cv2.resize(cv2.imread(image_path), (config.cropped_size, config.cropped_size)).astype(np.float32) / 255.
        image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))

        # read and process the mask
        image_mask = np.load(mask_path, allow_pickle=True)
        image_mask = image_mask[..., None].astype(np.float32)
        image_mask = image_mask.transpose(2, 0, 1)
        image_mask_bn = np.zeros_like(image_mask)
        image_mask_bn[np.where(image_mask != 0)] = 1.
        image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]).to(self.device))

        # load and normalize the landmark
        landmark = np.load(landmark_path).astype(np.float32)
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
        landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))

        # resize and create batch
        images = torch.cat(images, dim=0)
        images = F.interpolate(images, [self.image_size, self.image_size]) # <224, 224>
        image_masks = torch.cat(image_masks, dim=0)
        image_masks = F.interpolate(image_masks, [self.image_size, self.image_size])
        landmarks = torch.cat(landmarks, dim=0)
        
        # optimize
        single_params = self.optimize(images, landmarks, image_masks, savefolder)
        if self.config.save_all_output:
            self.render.save_obj(filename=os.path.join(savefolder, image_name+'.obj'),
                                 vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
                                 textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
                                 )
        np.save(os.path.join(savefolder, image_name+'.npy'), single_params)


def read_ldmk_file(ldmk_path, img_path):
	if not os.path.exists(ldmk_path):
		input_img = io.imread(img_path)
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
		ldmk = fa.get_landmarks(input_img)[0].astype(np.float64)
		np.save(ldmk_path, ldmk)
	else:
		ldmk = np.load(ldmk_path)

	return ldmk


def read_mask_file(mask_path, img_path):
	if not os.path.exists(mask_path):
		input_img = cv2.imread(img_path)
		input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
		mask = get_parsing_map(input_img)
		np.save(mask_path, mask)
	else:
		mask = np.load(mask_path)

	return mask


if __name__ == '__main__':
	device = 'cuda'

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_ind', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=5000)
	args = parser.parse_args()

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

	config['img_dir']  = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/synthetic_dataset/{}/origin_image'.format(config['cropped_size'])
	config['ldmk_dir'] = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/synthetic_dataset/{}/ldmk'.format(config['cropped_size'])
	config['mask_dir'] = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/synthetic_dataset/{}/mask'.format(config['cropped_size'])
	config['save_dir'] = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/synthetic_dataset/{}/3DMM_params'.format(config['cropped_size'])

	config = util.dict2obj(config)
	util.check_mkdir(config.save_dir)

	img_list = os.listdir(config.img_dir)[args.batch_ind * args.batch_size : (args.batch_ind+1) * args.batch_size]
	for img_ind in tqdm(range(len(img_list))):
		img_name = img_list[img_ind]
		img_path = os.path.join(config.img_dir, img_name)

		# check for landmark and mask ()
		img_prefix = img_name[:-4]
		ldmk_save_path = os.path.join(config.ldmk_dir, img_prefix+'_ldmk.npy')
		ldmk = read_ldmk_file(ldmk_save_path, img_path)
		mask_save_path = os.path.join(config.mask_dir, img_prefix+'_mask.npy')
		mask = read_mask_file(mask_save_path, img_path)

		config.batch_size = 1
		fitting = PhotometricFitting(config, device=device)
		fitting.run(img_path, ldmk_save_path, mask_save_path)