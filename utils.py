# from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from segment_model.model import BiSeNet

from pdb import set_trace as ST


# list of torch tensor --> concat numpy array
def latent_list2tensor(tensor_list):
    assert isinstance(tensor_list, list), 'list of pytorch tensor expected!'

    tensor_list_tmp = []
    for t in tensor_list:
        batch_size = t.shape[0]
        if batch_size == 1: # squeeze will kill the batch dimension
            tensor_list_tmp.append(t.squeeze().unsqueeze(0).cpu().numpy())
        else:
            tensor_list_tmp.append(t.squeeze().cpu().numpy())

    return np.concatenate(tensor_list_tmp, axis=1)


# concat numpy array --> list of torch tensor
def latent_tensor2list(latent_tensor):
    # assert len(latent_tensor.shape) == 2 and latent_tensor.shape[0] == 1, 'only tensors with shape [1, dim] are accepted'
    assert len(latent_tensor.shape) == 2, 'only tensors with shape [n_batch, dim] are accepted'

    _, n_dim = latent_tensor.shape
    if n_dim == 9088:
        dim_repeat_dict = {512:15, 256:3, 128:3, 64:3, 32:2}
    elif n_dim == 8704:
        dim_repeat_dict = {512:15, 256:3, 128:2}
    else:
        raise ValueError('Unrecognized style code length')

    tensor_list = []
    
    start_ind = 0
    for n_dim in dim_repeat_dict.keys():
        n_repeat = dim_repeat_dict[n_dim]
        for i in range(n_repeat):
            tensor_tmp = latent_tensor[:, start_ind+n_dim*i : start_ind+n_dim*(i+1)].copy().squeeze()
            tensor_tmp = torch.from_numpy(tensor_tmp).view([-1, 1, n_dim, 1, 1]).float().cuda() # automatic fill up the batch dimension
            tensor_list.append(tensor_tmp)

        start_ind += n_dim * n_repeat

    return tensor_list


def latent_interpolate(latent_code, w, b, latent_mode='style', end_point=1.0, steps=2):
    assert isinstance(w, np.ndarray) and isinstance(b, np.ndarray), 'Convert w and b to numpy.ndarray before interpolation!'

    if latent_mode == 'style':
        latent_code = latent_list2tensor(latent_code)
    else:
        latent_code = latent_code.cpu().numpy()

    assert isinstance(latent_code, np.ndarray), 'The latent code should have been converted to numpy.ndarray!'

    # interpolate in the latent space
    dist = (latent_code.dot(w.T) + b) / np.linalg.norm(w)
    linspace = np.linspace(-end_point, end_point, steps)
    linspace = linspace - dist # set the start point as the most negative sample
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    latent_code = np.tile(latent_code, (steps, 1)) + linspace * (w / np.linalg.norm(w))

    # convert the layout back
    if latent_mode == 'style': # tensor to list
        latent_code_list = [latent_tensor2list(latent_code[i:i+1, :]) for i in range(steps)]
    else: # numpy to cuda tensor
        latent_code_list = [torch.from_numpy(latent_code[i:i+1, :]).cuda() for i in range(steps)]

    return latent_code_list


def check_mkdir(path):
    if not os.path.exists(path):
        print('making %s' % path)
        os.makedirs(path)


def get_parsing_map(input_im, target_region_name='face', segnet=None, 
                    segnet_ckpt_path='/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/segment_model/cp/79999_iter.pth'):

    im_h, im_w, _ = input_im.shape

    if segnet is None:
        segnet = BiSeNet(n_classes=19).cuda()
        segnet.load_state_dict(torch.load(segnet_ckpt_path))
        segnet.eval()

    pre_process = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # scale the output tensor to [0, 1]
    if isinstance(input_im, np.ndarray):
        im = transforms.ToTensor()(input_im).clone().cuda()
        #input_im = torch.from_numpy(input_im).float().cuda()
    else:
        im = input_im.clone().cuda()

    min_val = float(im.min())
    max_val = float(im.max())
    im.clamp_(min=min_val, max=max_val)
    im.add_(-min_val).div_(max_val - min_val + 1e-6)

    # resize and normalize
    if len(im.shape) == 3:
        im = im.unsqueeze(0)
    im = F.interpolate(im, 512) # shape of input has to be [N, C, H, W], so for the output
    im = pre_process(im.squeeze(0)).cuda()

    # parse
    out = segnet(im.unsqueeze(0))[0]
    parsing_map = out.squeeze(0).cpu().detach().numpy().argmax(0)

    # obtain the binary mask accroding to the name of the target region
    attr_names = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                      'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    if target_region_name == 'face':
        cls_name = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose', 'mouth', 'u_lip', 'l_lip']
    else:
        cls_name == attr_name

    binary_mask = np.zeros((parsing_map.shape[0], parsing_map.shape[1]))
    for name in cls_name:
        attr_ind = attr_names.index(name)
        binary_mask[parsing_map == attr_ind] = 1

    binary_mask = cv2.resize(binary_mask, (im_h, im_w))

    return binary_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """ Resets the statistics of previous values. """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Add new value.

        Args:
            val (float): Value to add
            n (int): Count the value n times. Default is 1
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TensorBoardLogger(SummaryWriter):
    """ Writes entries directly to event files in the logdir to be consumed by TensorBoard.

    The logger keeps track of scalar values, allowing to easily log either the last value or the average value.

    Args:
        log_dir (str): The directory in which the log files will be written to
    """

    def __init__(self, log_dir=None):
        super(TensorBoardLogger, self).__init__(log_dir)
        self.__tb_logger = SummaryWriter(log_dir) if log_dir is not None else None
        self.log_dict = {}

    def reset(self, prefix=None):
        """ Resets all saved scalars and description prefix.

        Args:
            prefix (str, optional): The logger's prefix description used when printing the logger status
        """
        self.prefix = prefix
        self.log_dict.clear()

    def update(self, category='losses', **kwargs):
        """ Add named scalar values to the logger. If a scalar with the same name already exists, the new value will
        be associated with it.

        Args:
            category (str): The scalar category that will be concatenated with the main tag
            **kwargs: Named scalar values to be added to the logger.
        """
        if category not in self.log_dict:
            self.log_dict[category] = {}
        category_dict = self.log_dict[category]
        for key, val in kwargs.items():
            if key not in category_dict:
                category_dict[key] = AverageMeter()
            category_dict[key].update(val)

    def log_scalars_val(self, main_tag, global_step=None):
        """ Log the last value of all scalars.

        Args:
            main_tag (str): The parent name for the tags
            global_step (int, optional): Global step value to record
        """
        if self.__tb_logger is not None:
            for category, category_dict in self.log_dict.items():
                val_dict = {k: v.val for k, v in category_dict.items()}
                self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)

    def log_scalars_avg(self, main_tag, global_step=None):
        """ Log the average value of all scalars.

        Args:
            main_tag (str): The parent name for the tags
            global_step (int, optional): Global step value to record
        """
        if self.__tb_logger is not None:
            for category, category_dict in self.log_dict.items():
                val_dict = {k: v.avg for k, v in category_dict.items()}
                self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)

    def log_image(self, tag, img_tensor, global_step=None):
        """ Add an image tensor to the log.

        Args:
            tag (str): Name identifier for the image
            img_tensor (torch.Tensor): The image tensor to log
            global_step (int, optional): Global step value to record
        """
        if self.__tb_logger is not None:
            self.__tb_logger.add_image(tag, img_tensor, global_step)

    def __str__(self):
        desc = '' if self.prefix is None else self.prefix
        for category, category_dict in self.log_dict.items():
            desc += '\n{}: \n'.format(category)
            for key, log in category_dict.items():
                desc += ' {}: {:.4f} ({:.4f})\n '.format(key, log.val, log.avg)
            desc += '\n '

        return desc


def deTesnsor(x, norm):
    if norm:
        x = x / 2 + 0.5
    x *= 255
    x = x.permute((1,2,0))
    x = x.numpy().astype(np.uint8)

    return x
