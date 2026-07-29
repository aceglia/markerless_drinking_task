import cv2
from rtmlib import Wholebody, draw_skeleton, Body, Wholebody3d

import os
os.environ["YOLO_VERBOSE"] = 'False'
import numpy as np
from ultralytics import YOLO

device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style (required for animals), False for mmpose-style
wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend,
                       device=device)

start = 0
img = cv2.imread(color_image_path)
keypoints, scores = wholebody(img)
img = draw_skeleton(img, keypoints[0:1], scores, kpt_thr=0.1)
cv2.imwrite(color_image_path + '_annotated.png', img)