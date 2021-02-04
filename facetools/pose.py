import onnx
import cv2
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from .utils.common import BBox
from PIL import Image
# import estimator
import torch
from .architectures.hopenet import Hopenet, HopenetLite
from .utils.pose_utils import draw_axis

class hopenet_estimator:
    def __init__(self, backbone):
        this_path = os.path.dirname(os.path.abspath(__file__))
        if backbone == "resnet":
            self.model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            weight_path = os.path.join(this_path, "checkpoints/hopenet_robust_alpha1.pkl")
            self.model.load_state_dict(torch.load(weight_path))
        elif backbone == "shufflenet":
            self.model = HopenetLite()
            weight_path = os.path.join(this_path, "checkpoints/hopenet_lite.pkl")
            self.model.load_state_dict(torch.load(weight_path), strict=False)
        else:
            raise NotImplementedError("UNKOWN BACKBONE: {}".format(backbone))
        self.model.eval()
        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor)
        if torch.cuda.is_available():
            self.model.cuda()
            self.idx_tensor = self.idx_tensor.cuda()
        # setup the parameters
        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224), 
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                        
    def forward(self, orig_image, boxes):
        poses = []
        test_faces = []
        out_size = 512
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            img=orig_image.copy()
            height,width,_=img.shape
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2) 
            w = x2 - x1
            h = y2 - y1
            l = max(w, h)
            edx = max(0, l - w)
            edy = max(0, l - h)
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, 0, int(edy), 0, int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)    
            cropped_face = Image.fromarray(cropped_face)
            test_face = self.transformations(cropped_face)
            test_face.unsqueeze_(0)
            test_faces.append(test_face)
        
        if len(test_faces) == 0:
            return np.array([])
    
        with torch.no_grad():
            test_faces = torch.cat(test_faces, 0)
            test_faces = test_faces.cuda() if torch.cuda.is_available() else test_faces
            yaws, pitchs, rolls = self.model(test_faces)
            yaw_predicted = F.softmax(yaws, 1)
            pitch_predicted = F.softmax(pitchs, 1)
            roll_predicted = F.softmax(rolls, 1)
            
        
        for i in range(yaw_predicted.shape[0]):
            yaw, pitch, roll = yaw_predicted[i], pitch_predicted[i], roll_predicted[i]
            # Get continuous predictions in degrees.
            yaw = torch.sum(yaw.data * self.idx_tensor) * 3 - 99
            pitch = torch.sum(pitch.data * self.idx_tensor) * 3 - 99
            roll = torch.sum(roll.data * self.idx_tensor) * 3 - 99
            poses.append([yaw.cpu().numpy(), pitch.cpu().numpy(), roll.cpu().numpy()])
            
        return np.array(poses)


    def vis_poses(self, image, boxes, poses):
        for box, pose in zip(boxes, poses):
            yaw_predicted, pitch_predicted, roll_predicted = pose[0], pose[1], pose[2]
            x_min = box[0]
            x_max = box[2]
            y_min = box[1]
            y_max = box[3]
            bbox_height = y_max - y_min
            image = draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/4)
        return image