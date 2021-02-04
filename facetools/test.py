import cv2
from detect import onnx_detector, retinaface_detector
from align import onnx_kp_detector, mobile_kp_detector, fan_kp_detector, facemesh_detector, hrnet_detector
from track import tracker
from parse import bisenet_parser, ehanet_parser
from feature_extract import arcface_extractor
from pose import hopenet_estimator
import time

import numpy as np
from utils.arcface_utils import face_align
from utils.pose_utils import draw_axis

cap = cv2.VideoCapture(0)  # capture from camera
myDetector = onnx_detector()#retinaface_detector("mobilenet")#
myExtractor = None#arcface_extractor("mobilenet")#
myKPD = None#facemesh_detector()#fan_kp_detector()#hrnet_detector()#onnx_kp_detector()#mobile_kp_detector()#
myTracker = tracker()
# parse size = [320, 512] is recommend
myParser = ehanet_parser(320)#bisenet_parser()#None#
myposer = None#hopenet_estimator("shufflenet")#

if myExtractor is not None:
    myPhoto = cv2.imread("zyh.jpg")
    _, _, myLDM = myDetector.forward(myPhoto, threshold=0.5)
    myFeatures = myExtractor.forward(myPhoto, myLDM)
    myFeatures = myFeatures.transpose((1, 0))

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("no img")
        break
    #start = time.time()
    boxes, probs, landmarks5 = myDetector.forward(orig_image, threshold=0.5)
    boxes, landmarks, masks = myTracker.forward(orig_image, boxes, landmarks5, myKPD, myParser)
    if myExtractor is not None:
        face_features = myExtractor.forward(orig_image, landmarks5)
        similarities = np.dot(face_features, myFeatures)
    
    # drawing results
    if myposer is not None:
        poses = myposer.forward(orig_image, boxes)
        orig_image = myposer.vis_poses(orig_image, boxes, poses)
    if masks is not None:
        mask_one = myParser.vis_maps(orig_image, boxes, masks)
        orig_image = cv2.addWeighted(orig_image, 0.7, mask_one, 0.3, 0)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
        if myExtractor is not None:
            simi = similarities[i][0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            if simi > 0.6:
                label = "zyh"
            else:
                label = "Unkown"
            cv2.putText(orig_image, label, (box[0], box[1]-10), font, 1.0, (255, 0, 255), 2)
            #cv2.putText(orig_image, "{:.3f}".format(simi), (box[0], box[1]-10), font, 1.0, (255, 0, 255), 2)
    if landmarks is not None:
        for landmark in landmarks:
            for x, y in landmark:
                cv2.circle(orig_image, (int(x), int(y)), 1, (0,255,0), -1)
    cv2.imshow('annotated', orig_image)
    #end = time.time()
    #print(end-start)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()