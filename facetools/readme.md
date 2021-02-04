# facetools

## pre-request

- onnx `pip install onnx`
- onnxruntime `pip install onnxruntime`
- pytorch
- opencv-python
- PIL
- ...

## face detector

- [x] [onnx detector]((<https://github.com/cunjian/pytorch_face_landmark>)) (without 5 key points, real-time on CPU)
- [x] retinaface (with 5 key points, backbone=[mobilenet, resnet50])

returned bbox is formatted as [left, top, right, bottom]

## face alignment

`landmarks.astype(np.int)` is needed for plotting, here we return float value is for tracker

- **68 landmark detectors**

- [x] [onnx landmark detector]((<https://github.com/cunjian/pytorch_face_landmark>)) (real-time on CPU, inference only on a single image)
- [x] [mobilenet landmark detector]((<https://github.com/cunjian/pytorch_face_landmark>)) (real-time on GPU, support batch inference)
- [x] [FAN](<https://github.com/1adrianb/face-alignment>) 

- **98 landmark detector**

- [x] [HRNet](<https://github.com/HRNet/HRNet-Facial-Landmark-Detection>)

- **468 landmark detector**

- [x] [facemesh](<https://github.com/thepowerfuldeez/facemesh.pytorch>) (dense landmarks in 3D, real-time on GPU (maybe on CPU), support batch inference)

## face tracker

- [x] tracker

  **NOT** a trace algorithm, used to smooth bounding box and landmark locations

## face segmentation

- [x] [BiSeNet](<https://github.com/zllrunning/face-parsing.PyTorch>) (around 10 fps on 2080ti)

- [ ] [EHANet](<https://github.com/JACKYLUO1991/FaceParsing>)

  |                 |             |             |
  | --------------- | ----------- | ----------- |
  | Label list      |             |             |
  | 0: 'background' | 1: 'skin'   | 2: 'l_brow' |
  | 3: 'r_brow'     | 4: 'l_eye'  | 5: 'r_eye'  |
  | 6: 'eye_g'      | 7: 'l_ear'  | 8: 'r_ear'  |
  | 9: 'ear_r'      | 10: 'nose'  | 11: 'mouth' |
  | 12: 'u_lip'     | 13: 'l_lip' | 14: 'neck'  |
  | 15: 'neck_l'    | 16: 'cloth' | 17: 'hair'  |
  | 18: 'hat'       |             |             |

## face recognition

- [x] arcface (w/ or w/o 5 key points align, backbone=[mobilenet, resnet50])

  It is suggested that to extract features after face alignment

## face pose

- [x] [hopenet](<https://github.com/natanielruiz/deep-head-pose>) (backbone=[shufflenet, resnet50])

  It is said that "You can run the shufflenet network on CPU (i7-8700 six cores) with **35 FPS** or GPU (RTX 2070) with **130 FPS**"