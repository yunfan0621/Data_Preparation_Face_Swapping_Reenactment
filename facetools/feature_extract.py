import torch
import numpy as np
import cv2
import os
from .architectures.arcface import Backbone, MobiFaceNet, remove_module_dict
from .utils.arcface_utils import face_align

class arcface_extractor:
    def __init__(self, backbone):
        weights_path = None
        this_path = os.path.dirname(os.path.abspath(__file__))
        if backbone == "mobilenet":
            weights_path = os.path.join(this_path, "checkpoints/arcface_mb.pth")
            self.model = MobiFaceNet()
            self.model.load_state_dict(remove_module_dict(torch.load(weights_path)))
        elif backbone == "resnet":
            weights_path = os.path.join(this_path, "checkpoints/arcface_r50.pth")
            self.model = Backbone(50)
            self.model.load_state_dict(remove_module_dict(torch.load(weights_path)))
        else:
            raise NotImplementedError("UNOKOWN BACKBONE: {}".format(backbone))
        
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        
    
    def forward(self, orig_image, landmarks):
        aligned_faces = []
        if landmarks is None:
            landmarks = [None]
        for ldm in landmarks:
            aligned_face = face_align(orig_image, None, ldm, image_size="112, 112")
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            aligned_faces.append(aligned_face)

        image_numpy = 2 * (np.array(aligned_faces)/255.0 - 0.5)
        image_tensor = torch.Tensor(image_numpy).cuda()
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        with torch.no_grad():
            features = self.model(image_tensor)
        features = features.cpu().numpy()
        return features