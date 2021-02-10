import os
import cv2
import numpy as np
from utils import get_parsing_map

from pdb import set_trace as ST

if __name__ == '__main__':
	img_dir = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/FFHQ/'
	seg_dir = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/FFHQ_seg/'

	# face mask prediction from image
	img_name = '00000.png'
	input_img = cv2.imread(os.path.join(img_dir, img_name))
	input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
	pred_mask = get_parsing_map(input_img)

	# face mask read from disk
	mask_name = '00000.npy'
	input_mask = np.load(os.path.join(seg_dir, mask_name))

	ST()