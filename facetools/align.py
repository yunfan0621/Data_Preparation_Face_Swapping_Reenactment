import onnx
import cv2
import os
import numpy as np
# onnx runtime
import onnxruntime as ort
# import libraries for landmark
from .utils.common import BBox,drawLandmark,drawLandmark_multiple
from PIL import Image
import torchvision.transforms as transforms
# import mobilenet keypoint detector
import torch
from .architectures.MobileNet import MobileNet_GDConv
from .architectures.fan import FAN
from .architectures.facemesh import FaceMesh
from .architectures.hrnet import HRNet
from .utils.hrnet_utils import hflip_face_landmarks_98pts, LandmarksHeatMapEncoder

class onnx_kp_detector:
    def __init__(self):
        self.input_size = 56
        this_path = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(this_path, "checkpoints/landmark_detection_56_se_external.onnx")
        self.ort_session_landmark = ort.InferenceSession(onnx_path)
        # setup the parameters
        self.resize = transforms.Resize([self.input_size, self.input_size])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, orig_image, boxes):
        landmarks = []
        out_size = self.input_size
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            img=orig_image.copy()
            height,width,_=img.shape
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)    
            cropped_face = Image.fromarray(cropped_face)
            test_face = self.resize(cropped_face)
            test_face = self.to_tensor(test_face)
            test_face = self.normalize(test_face)
            test_face.unsqueeze_(0)
                
            ort_inputs = {self.ort_session_landmark.get_inputs()[0].name: self.to_numpy(test_face)}
            ort_outs = self.ort_session_landmark.run(None, ort_inputs)
            landmark = ort_outs[0]
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark)
            landmarks.append(landmark)
        return np.array(landmarks)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class mobile_kp_detector:
    def __init__(self):
        self.input_size = 224
        self.model = MobileNet_GDConv(136)
        this_path = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(this_path, "checkpoints/mobilenet_224_model_best_gdconv.pth.tar")
        self.model.load_state_dict(torch.load(weight_path)['state_dict'])
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        # setup the parameters
        self.resize = transforms.Resize([self.input_size, self.input_size])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, orig_image, boxes):
        landmarks = []
        test_faces = []
        new_boxes = []
        out_size = self.input_size
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            img=orig_image.copy()
            height,width,_=img.shape
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)    
            cropped_face = Image.fromarray(cropped_face)
            test_face = self.resize(cropped_face)
            test_face = self.to_tensor(test_face)
            test_face = self.normalize(test_face)
            test_face.unsqueeze_(0)
            test_faces.append(test_face)
            new_boxes.append(new_bbox)
        if len(test_faces) == 0:
            return np.array(landmarks)
        test_faces = torch.cat(test_faces, 0)
        with torch.no_grad():
            test_faces = test_faces.cuda() if torch.cuda.is_available() else test_faces
            landmarks_ = self.model(test_faces).cpu().numpy()
        for i in range(landmarks_.shape[0]):
            landmark = landmarks_[i]
            new_bbox = new_boxes[i]
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark)
            landmarks.append(landmark)
        return np.array(landmarks)
        

class fan_kp_detector:
    def __init__(self):
        self.input_size = 256
        self.model = FAN()
        this_path = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(this_path, "checkpoints/2DFAN.pth")
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        self.to_landmark = LandmarksHeatMapEncoder()
        if torch.cuda.is_available():
            self.model.cuda()
            self.to_landmark.cuda()
        
        self.to_tensor = transforms.ToTensor()

    def forward(self, orig_image, boxes):
        landmarks = []
        test_faces = []
        new_boxes = []
        out_size = self.input_size
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            img=orig_image.copy()
            height,width,_=img.shape
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)   
            cropped_face = self.to_tensor(cropped_face)
            test_faces.append(cropped_face.unsqueeze_(0))
            new_boxes.append(new_bbox)

        if len(test_faces) == 0:
            return np.array(landmarks)
        test_faces = torch.cat(test_faces, 0)
        with torch.no_grad():
            test_faces = test_faces.cuda() if torch.cuda.is_available() else test_faces
            outs = self.model(test_faces)[-1]
            landmarks_ = self.to_landmark(outs).cpu()
        
        for i in range(landmarks_.shape[0]):
            landmark = landmarks_[i]
            new_bbox = new_boxes[i]
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark[:,:2])
            landmarks.append(landmark)
        return np.array(landmarks)

class facemesh_detector:
    def __init__(self):
        self.input_size = 192
        self.model = FaceMesh()
        this_path = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(this_path, "checkpoints/facemesh.pth")
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        # setup the parameters
        self.resize = transforms.Resize([self.input_size, self.input_size])

    def forward(self, orig_image, boxes):
        landmarks = []
        test_faces = []
        new_boxes = []
        out_size = self.input_size
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            img=orig_image.copy()
            height,width,_=img.shape
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)   
            test_face = cv2.resize(cropped_face, (out_size, out_size))
            test_faces.append(test_face)
            new_boxes.append(new_bbox)
        if len(test_faces) == 0:
            return np.array(landmarks)
        test_faces = np.array(test_faces)
        landmarks_ = self.model.predict_on_batch(test_faces)[0].cpu().numpy()
        for i in range(landmarks_.shape[0]):
            landmark = landmarks_[i]
            new_bbox = new_boxes[i]
            landmark = landmark.reshape(-1,3)/192
            landmark = new_bbox.reprojectLandmark(landmark[:,:2])
            landmarks.append(landmark)
        return np.array(landmarks)

class hrnet_detector:
    def __init__(self):
        self.input_size = 256
        self.model = HRNet()
        this_path = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(this_path, "checkpoints/hrnet.pth")
        self.model.load_state_dict(torch.load(weight_path)["state_dict"])
        self.model.eval()
        self.to_landmark = LandmarksHeatMapEncoder()
        if torch.cuda.is_available():
            self.model.cuda()
            self.to_landmark.cuda()
        # setup the parameters
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, orig_image, boxes):
        landmarks = []
        test_faces = []
        new_boxes = []
        out_size = self.input_size
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            img=orig_image.copy()
            height,width,_=img.shape
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)   
            cropped_face = self.to_tensor(cropped_face)
            test_face = self.normalize(cropped_face)
            test_faces.append(test_face.unsqueeze_(0))
            new_boxes.append(new_bbox)

        if len(test_faces) == 0:
            return np.array(landmarks)
        test_faces = torch.cat(test_faces, 0)
        with torch.no_grad():
            test_faces = test_faces.cuda() if torch.cuda.is_available() else test_faces
            outs = self.model(test_faces)
            landmarks_ = self.to_landmark(outs).cpu()
        
        for i in range(landmarks_.shape[0]):
            landmark = landmarks_[i]
            new_bbox = new_boxes[i]
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark[:,:2])
            landmarks.append(landmark)
        return np.array(landmarks)