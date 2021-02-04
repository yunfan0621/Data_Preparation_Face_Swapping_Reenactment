import onnx
import cv2
import os
import numpy as np
import onnxruntime as ort
import torch
from .utils  import box_utils_numpy as box_utils
from .architectures.retinaface import RetinaFace
from .utils.retinaface_utils import decode, decode_landm, py_cpu_nms, PriorBox

class onnx_detector:
    def __init__(self):
        this_path = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(this_path, "checkpoints/version-RFB-320.onnx")
        #predictor = onnx.load(onnx_path)
        #onnx.checker.check_model(predictor)
        #onnx.helper.printable_graph(predictor.graph)
        #predictor = backend.prepare(predictor, device="CPU")  # default CPU

        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
    
    def forward(self, orig_image, threshold=0.7, box_scale=1.2):
        height,width,_ = orig_image.shape
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, _, probs = self.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        new_bbox = []
        for box in boxes:
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(max([w, h])*box_scale)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = cx + size//2
            y1 = cy - size//2
            y2 = cy + size//2
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            # left, top, right, bottom
            new_bbox.append(list(map(int, [x1, y1, x2, y2])))
        new_bbox = np.array(new_bbox)
        probs = np.array(probs)
        return new_bbox, probs, None
    
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # face detection setting
    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

class retinaface_detector:
    def __init__(self, backbone):
        weights_path = None
        this_path = os.path.dirname(os.path.abspath(__file__))
        if backbone == "mobilenet":
            weights_path =  os.path.join(this_path, "checkpoints/retinaface_mb25.pth")
            self.resolution = 320
            self.model = RetinaFace(encoder="mobilenet")
        else:
            weights_path =  os.path.join(this_path, "checkpoints/retinaface_r50.pth")
            self.resolution = 640
            self.model = RetinaFace(encoder="resnet")
        
        priorbox = PriorBox(self.model.cfg, image_size=(self.resolution, self.resolution))
        priors = priorbox.forward()
        self.prior_data = priors.data
        self.model = self.load_model(self.model, weights_path, True)
        if torch.cuda.is_available():
            self.prior_data = self.prior_data.cuda()
            self.model.cuda()
        self.model.eval()
    
    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        #print('Missing keys:{}'.format(len(missing_keys)))
        #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        #print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        #print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        #print('Loading pretrained model from {}'.format(pretrained_path)
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    
    def forward(self, orig_image, threshold=0.5, box_scale=1.2):
        img = orig_image
        im_height, im_width, _ = orig_image.shape
        long_length = max(im_height, im_width)
        resize = self.resolution/long_length
        edx = long_length-im_width
        edy = long_length-im_height
        img = cv2.copyMakeBorder(orig_image, 0, int(edy), 0, int(edx), cv2.BORDER_CONSTANT, 0)            
        img = cv2.resize(img, (self.resolution, self.resolution))

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale1 = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0]])
        img = img.astype(np.float32)
        img -= (104., 117., 123.)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
            scale = scale.cuda()
            scale1 = scale1.cuda()
        loc, conf, landms = self.model(img)
        boxes = decode(loc.data.squeeze(0), self.prior_data, self.model.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.model.cfg['variance'])
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:1000]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.3)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:500, :]
        landms = landms[:500, :]

        dets = np.concatenate((dets, landms), axis=1)
        new_boxes = []
        probs = []
        landmarks = []
        for i in range(len(dets)):
            box = dets[i]
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(max([w, h])*box_scale)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = cx + size//2
            y1 = cy - size//2
            y2 = cy + size//2
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_width, x2)
            y2 = min(im_height, y2)
            # left, top, right, bottom
            new_boxes.append(list(map(int, [x1, y1, x2, y2])))
            probs.append(box[4])
            landmarks.append(landms[i].reshape((5,2)))
        new_boxes = np.array(new_boxes)
        probs = np.array(probs)
        landmarks = np.array(landmarks)
        return new_boxes, probs, landmarks

