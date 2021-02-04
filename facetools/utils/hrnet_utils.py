def hflip_face_landmarks_98pts(landmarks, width=1):
    """ Horizontal flip 98 points landmarks.
    Args:
        landmarks (np.array): Landmarks points of shape (98, 2)
        width (int): The width of the correspondign image
    Returns:
        np.array: Horizontally flipped landmarks.
    """
    assert landmarks.shape[0] == 98
    landmarks = landmarks.copy()

    # Invert X coordinates
    for p in landmarks:
        p[0] = width - p[0]

    # Jaw
    right_jaw, left_jaw = list(range(0, 16)), list(range(32, 16, -1))
    landmarks[right_jaw + left_jaw] = landmarks[left_jaw + right_jaw]

    # Eyebrows
    right_brow, left_brow = list(range(33, 42)), list(range(46, 41, -1)) + list(range(50, 46, -1))
    landmarks[right_brow + left_brow] = landmarks[left_brow + right_brow]

    # Nose
    right_nostril, left_nostril = list(range(55, 57)), list(range(59, 57, -1))
    landmarks[right_nostril + left_nostril] = landmarks[left_nostril + right_nostril]

    # Eyes
    right_eye, left_eye = list(range(60, 68)) + [96], [72, 71, 70, 69, 68, 75, 74, 73, 97]
    landmarks[right_eye + left_eye] = landmarks[left_eye + right_eye]

    # Mouth outer
    mouth_out_right, mouth_out_left = [76, 77, 78, 87, 86], [82, 81, 80, 83, 84]
    landmarks[mouth_out_right + mouth_out_left] = landmarks[mouth_out_left + mouth_out_right]

    # Mouth inner
    mouth_in_right, mouth_in_left = [88, 89, 95], [92, 91, 93]
    landmarks[mouth_in_right + mouth_in_left] = landmarks[mouth_in_left + mouth_in_right]

    return landmarks

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

def filter_landmarks(landmarks, threshold=0.5):
    """ Filter landmarks feature map activations by threshold.
    Args:
        landmarks (torch.Tensor): Landmarks feature map of shape (B, C, H, W)
        threshold (float): Filtering threshold
    Returns:
        torch.Tensor: Filtered landmarks feature map of shape (B, C, H, W)
    """
    landmarks_min = landmarks.view(landmarks.shape[:2] + (-1,)).min(2)[0].view(landmarks.shape[:2] + (1, 1))
    landmarks_max = landmarks.view(landmarks.shape[:2] + (-1,)).max(2)[0].view(landmarks.shape[:2] + (1, 1))
    landmarks = (landmarks - landmarks_min) / (landmarks_max - landmarks_min)
    # landmarks.pow_(2)
    landmarks[landmarks < threshold] = 0.0

    return landmarks


class LandmarksHeatMapEncoder(nn.Module):
    """ Encodes landmarks heatmap into a landmarks vector of points.
    Args:
        size (int or sequence of int): the size of the landmarks heat map (height, width)
    """
    def __init__(self, size=64):
        super(LandmarksHeatMapEncoder, self).__init__()
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        size = (size, size) if isinstance(size, int) else size
        y_indices, x_indices = torch.meshgrid(torch.arange(0., size[1]), torch.arange(0., size[0]))
        self.register_buffer('x_indices', x_indices.add(0.5) / size[1])
        self.register_buffer('y_indices', y_indices.add(0.5) / size[0])

    def __call__(self, landmarks):
        """ Encode landmarks heatmap to landmarks points.
        Args:
            landmarks (torch.Tensor): Landmarks heatmap of shape (B, C, H, W)
        Returns:
            torch.Tensor: Encoded landmarks points of shape (B, C, 2).
        """
        landmarks = filter_landmarks(landmarks)
        w = landmarks.div(landmarks.view(landmarks.shape[:2] + (-1,)).sum(dim=2).view(landmarks.shape[:2] + (1, 1)))
        x = w * self.x_indices
        y = w * self.y_indices
        x = x.view(x.shape[:2] + (-1,)).sum(dim=2).unsqueeze(2)
        y = y.view(y.shape[:2] + (-1,)).sum(dim=2).unsqueeze(2)
        landmarks = torch.cat((x, y), dim=2)

        return landmarks