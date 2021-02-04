import os
import torch
from skimage import io
from skimage import color
import numpy as np
import cv2
import torchvision.transforms as transforms

class group_track():
    def __init__(self):
        self.old_frame = None
        self.previous_landmarks_set = None
        self.with_landmark = True
        self.thres=1
        self.alpha=0.95 # landmark smooth
        self.iou_thres=0.5

    def calculate(self, img, now_landmarks_set):
        if self.previous_landmarks_set is None or self.previous_landmarks_set.shape[0]==0:
            self.previous_landmarks_set=now_landmarks_set
            result = now_landmarks_set
        else:
            if self.previous_landmarks_set.shape[0]==0:
                return now_landmarks_set
            else:
                result=[]
                for i in range(now_landmarks_set.shape[0]):
                    not_in_flag = True
                    for j in range(self.previous_landmarks_set.shape[0]):
                        if self.iou(now_landmarks_set[i],self.previous_landmarks_set[j])>self.iou_thres:
                            result.append(self.smooth(now_landmarks_set[i],self.previous_landmarks_set[j]))
                            not_in_flag=False
                            break
                    if not_in_flag:
                        result.append(now_landmarks_set[i])
        result=np.array(result)
        self.previous_landmarks_set=result

        return result

    def iou(self,p_set0,p_set1):
        rec1=[np.min(p_set0[:,0]),np.min(p_set0[:,1]),np.max(p_set0[:,0]),np.max(p_set0[:,1])]
        rec2 = [np.min(p_set1[:, 0]), np.min(p_set1[:, 1]), np.max(p_set1[:, 0]), np.max(p_set1[:, 1])]
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
        # find the each edge of intersect rectangle
        x1 = max(rec1[0], rec2[0])
        y1 = max(rec1[1], rec2[1])
        x2 = min(rec1[2], rec2[2])
        y2 = min(rec1[3], rec2[3])
        # judge if there is an intersect
        intersect = max(0, x2 - x1) * max(0, y2 - y1)
        if intersect == 0 and sum_area == 0:
            return 0.0
        return intersect / (sum_area - intersect)

    def smooth(self,now_landmarks,previous_landmarks):
        result=[]
        for i in range(now_landmarks.shape[0]):
            dis = np.sqrt(np.square(now_landmarks[i][0] - previous_landmarks[i][0]) + np.square(now_landmarks[i][1] - previous_landmarks[i][1]))
            if dis < self.thres:
                result.append(previous_landmarks[i])
            else:
                result.append(self.do_moving_average(now_landmarks[i], previous_landmarks[i]))
        return np.array(result)

    def do_moving_average(self,p_now,p_previous):
        p=self.alpha*p_now+(1-self.alpha)*p_previous
        return p


class tracker():
    def __init__(self):
        self.trace = group_track()

        ###another thread should run detector in a slow way and update the track_box
        self.track_box=None
        self.previous_image=None

        self.diff_thres=5
        self.top_k = 10
        self.iou_thres=0.3
        self.alpha=0.5 # bbox smooth

    def forward(self, image, boxes, landmarks=None, kp_detector=None, parser=None):
        landmarks = landmarks
        masks = None
        if self.diff_frames(self.previous_image,image):
            self.previous_image=image
            boxes = self.judge_boxs(self.track_box, boxes)
        else:
            boxes=self.track_box
            self.previous_image = image

        if boxes.shape[0]>self.top_k:
            boxes=self.sort(boxes)

        boxes_return = np.array(boxes)
        self.track_box = boxes_return

        if kp_detector is not None:
            landmarks = kp_detector.forward(image, self.track_box.astype(np.int))
        if landmarks is not None:
            landmarks = self.trace.calculate(image, landmarks)
            landmarks = landmarks.astype(np.int)
        if parser is not None:
            masks = parser.forward(image, self.track_box.astype(np.int))
        return self.track_box.astype(np.int), landmarks, masks

    def diff_frames(self,previous_frame,image):
        if previous_frame is None:
            return True
        else:
            _diff = cv2.absdiff(previous_frame, image)
            diff=np.sum(_diff)/previous_frame.shape[0]/previous_frame.shape[1]/3.
            if diff>self.diff_thres:
                return True
            else:
                return False

    def sort(self,bboxes):
        if self.top_k >100:
            return bboxes
        area=[]
        for bbox in bboxes:

            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            area.append(bbox_height*bbox_width)
        area=np.array(area)

        picked=area.argsort()[-self.top_k:][::-1]
        sorted_bboxes=[bboxes[x] for x in picked]
        return np.array(sorted_bboxes)

    def judge_boxs(self,previuous_bboxs,now_bboxs):
        def iou(rec1, rec2):

            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            x1 = max(rec1[0], rec2[0])
            y1 = max(rec1[1], rec2[1])
            x2 = min(rec1[2], rec2[2])
            y2 = min(rec1[3], rec2[3])

            # judge if there is an intersect
            intersect =max(0,x2-x1) * max(0,y2-y1)
            if intersect == 0 and sum_area == 0:
                return 0.0
            return intersect / (sum_area - intersect)

        if previuous_bboxs is None:
            return now_bboxs

        result=[]

        for i in range(now_bboxs.shape[0]):
            contain = False
            for j in range(previuous_bboxs.shape[0]):
                if iou(now_bboxs[i], previuous_bboxs[j]) > self.iou_thres:
                    result.append(self.smooth(now_bboxs[i],previuous_bboxs[j]))
                    contain=True
                    break
            if not contain:
                result.append(now_bboxs[i])


        return np.array(result)

    def smooth(self,now_box,previous_box):
        return self.do_moving_average(now_box[:4], previous_box[:4])

    def do_moving_average(self,p_now,p_previous):
        p=self.alpha*p_now+(1-self.alpha)*p_previous
        return p