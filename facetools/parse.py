import onnx
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from .utils.common import BBox
from PIL import Image
# import parser
import time
import torch
from .architectures.BiSeNet import BiSeNet
from .architectures.ehanet import EHANet18, EHANet34

class bisenet_parser:
    def __init__(self, size=512):
        self.model = BiSeNet(19)
        this_path = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(this_path, "checkpoints/bisenet_512.pth")
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        # setup the parameters
        self.input_size = size
        self.resize = transforms.Resize([self.input_size, self.input_size])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                            [255, 0, 85], [255, 0, 170],
                            [0, 255, 0], [85, 255, 0], [170, 255, 0],
                            [0, 255, 85], [0, 255, 170],
                            [0, 0, 255], [85, 0, 255], [170, 0, 255],
                            [0, 85, 255], [0, 170, 255],
                            [255, 255, 0], [255, 255, 85], [255, 255, 170],
                            [255, 0, 255], [255, 85, 255], [255, 170, 255],
                            [0, 255, 255], [85, 255, 255], [170, 255, 255]]
                        
    def forward(self, orig_image, boxes):
        masks = []
        test_faces = []
        out_size = self.input_size
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
            test_face = self.resize(cropped_face)
            test_face = self.to_tensor(test_face)
            test_face = self.normalize(test_face)
            test_face.unsqueeze_(0)
            test_faces.append(test_face)
        
        if len(test_faces) == 0:
            return np.array([])
    
        with torch.no_grad():
            #start = time.time()
            test_faces = torch.cat(test_faces, 0)
            test_faces = test_faces.cuda() if torch.cuda.is_available() else test_faces
            out = self.model(test_faces)[0].cpu().numpy()
            #end = time.time()
            #print(end-start)
        
        for i in range(out.shape[0]):
            box = boxes[i, :]
            height,width,_=orig_image.shape
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
            size = int(max(w, h))
            parse = out[i]
            _,height,width=parse.shape
            parse = parse.argmax(0)[:, :, np.newaxis].astype(np.uint8)
            #parse = cv2.cvtColor(parse, cv2.COLOR_GRAY2BGR)   
            #print(parse.shape)
            #print(size)
            #print(height-edy, width-edx)
            parse = cv2.resize(parse, (size, size))
            new_h = int(min(h, parse.shape[0]))
            new_w = int(min(w, parse.shape[1]))
            masks.append(np.array(parse[:new_h, :new_w]))
        return np.array(masks)


    def vis_maps(self, image, boxes, masks):
        #vis_image = image.copy().astype(np.uint8)
        vis_parse = np.zeros(image.shape) + 255
        height,width,_=image.shape
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            
            vis_parsing_anno = masks[i].copy().astype(np.uint8)
            vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
            num_of_class = np.max(vis_parsing_anno)
            for pi in range(1, num_of_class + 1):
                index = np.where(vis_parsing_anno == pi)
                vis_parsing_anno_color[index[0], index[1], :] = self.part_colors[pi]
            vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
            new_h = min(y2-y1, vis_parsing_anno_color.shape[0])
            new_w = min(x2-x1, vis_parsing_anno_color.shape[1])
            vis_parse[y1:y1+new_h, x1:x1+new_w] = vis_parsing_anno_color[:new_h, :new_w]
        return vis_parse.astype(np.uint8)

class ehanet_parser:
    def __init__(self, depth=18, size=512):
        if depth == 18:
            self.model = EHANet18()
            this_path = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(this_path, "checkpoints/ehanet18.pth")
            self.model.load_state_dict(torch.load(weight_path))
        else:
            self.model = EHANet34()
            this_path = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(this_path, "checkpoints/ehanet34.pth")
            self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        # setup the parameters
        self.input_size = size
        self.resize = transforms.Resize([self.input_size, self.input_size])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                            [255, 0, 85], [255, 0, 170],
                            [0, 255, 0], [85, 255, 0], [170, 255, 0],
                            [0, 255, 85], [0, 255, 170],
                            [0, 0, 255], [85, 0, 255], [170, 0, 255],
                            [0, 85, 255], [0, 170, 255],
                            [255, 255, 0], [255, 255, 85], [255, 255, 170],
                            [255, 0, 255], [255, 85, 255], [255, 170, 255],
                            [0, 255, 255], [85, 255, 255], [170, 255, 255]]
                        
    def forward(self, orig_image, boxes, return_raw=False):
        masks = []
        test_faces = []
        out_size = self.input_size
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
            test_face = self.resize(cropped_face)
            test_face = self.to_tensor(test_face)
            test_face = self.normalize(test_face)
            test_face.unsqueeze_(0)
            test_faces.append(test_face)
        
        if len(test_faces) == 0:
            return np.array([])
    
        with torch.no_grad():
            #start = time.time()
            test_faces = torch.cat(test_faces, 0)
            test_faces = test_faces.cuda() if torch.cuda.is_available() else test_faces
            out = self.model(test_faces).cpu().numpy()
            #end = time.time()
            #print(end-start)
        
        for i in range(out.shape[0]):
            box = boxes[i, :]
            height,width,_=orig_image.shape
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
            size = int(max(w, h))
            parse = out[i]
            _,height,width=parse.shape
            parse = parse.argmax(0)[:, :, np.newaxis].astype(np.uint8)
            #parse = cv2.cvtColor(parse, cv2.COLOR_GRAY2BGR)   
            #print(parse.shape)
            #print(size)
            #print(height-edy, width-edx)
            parse = cv2.resize(parse, (size, size))
            new_h = int(min(h, parse.shape[0]))
            new_w = int(min(w, parse.shape[1]))
            masks.append(np.array(parse[:new_h, :new_w]))
        return np.array(masks)


    def vis_maps(self, image, boxes, masks):
        #vis_image = image.copy().astype(np.uint8)
        vis_parse = np.zeros(image.shape) + 255
        height,width,_=image.shape
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            
            vis_parsing_anno = masks[i].copy().astype(np.uint8)
            vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
            num_of_class = np.max(vis_parsing_anno)
            for pi in range(1, num_of_class + 1):
                index = np.where(vis_parsing_anno == pi)
                vis_parsing_anno_color[index[0], index[1], :] = self.part_colors[pi]
            vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
            new_h = min(y2-y1, vis_parsing_anno_color.shape[0])
            new_w = min(x2-x1, vis_parsing_anno_color.shape[1])
            vis_parse[y1:y1+new_h, x1:x1+new_w] = vis_parsing_anno_color[:new_h, :new_w]
        return vis_parse.astype(np.uint8)