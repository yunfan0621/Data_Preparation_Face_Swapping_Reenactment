import os
import numpy as np
import face_alignment
from skimage import io

from pdb import set_trace as ST

if __name__ == '__main__':
	data_dir  = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/3DMM_models/FLAME_Fitting/FFHQ/'

	# landmark prediction from image
	img_name = '00000.png'
	input_img = io.imread(os.path.join(data_dir, img_name))

	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	preds = fa.get_landmarks(input_img)[0]

	# landmarks read from disk
	ldmk_name = '00000.npy'
	input_ldmk = np.load(os.path.join(data_dir, ldmk_name))