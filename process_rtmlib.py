import cv2
from rtmlib import Wholebody, draw_skeleton, Body, Wholebody3d

import os
os.environ["YOLO_VERBOSE"] = 'False'
import numpy as np
from ultralytics import YOLO

# import matplotlib.pyplot as plt

# def show_glass_bbox(img, bboxes, cup_idx):
#     for boxes in bboxes:
#         boxes_ = boxes.boxes.xyxy.cpu().numpy()
#         scores = boxes.boxes.conf.cpu().numpy()
#         classes = boxes.boxes.cls.cpu().numpy()
#         for box, score, cls in zip(boxes_, scores, classes):
#             if cls != cup_idx:
#                 continue
#             x1, y1, x2, y2 = map(int, box)
#             box_score = score
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return score
    
# det = YOLO("yolov8n.pt")
# cup_idx = [i for i in det.names if det.names[i] == "cup"][0]
device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style (required for animals), False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend,
                       device=device)
# dir_path = "/mnt/c/Users/Usager/Documents/Amedeo/rgbd_ms_cime/videos/img_0"
dir_path = '/mnt/c/Users/Usager/Documents/Amedeo/rgbd_ms_cime/videos/test_CIME_LOC/20250422_162118'
dir_path = '/mnt/c/Users/Usager/Documents/Amedeo/rgbd_ms_cime/videos/test_lexie'


all_image = os.listdir(dir_path)
all_image = [os.path.join(dir_path, im) for im in all_image if (im.endswith(".png") and "color" in im)]
idxs = [int(os.path.basename(file).split("_")[1].split(".")[0]) for file in all_image]

idxs = sorted(idxs)
os.makedirs(dir_path + '/annotated', exist_ok=True)
os.makedirs(dir_path + '/annotated_glass_bb', exist_ok=True)
keypoints_mat = np.empty((len(all_image), 133, 4))
start = 0
for i, idx in enumerate(idxs[start:]):

    img = cv2.imread(os.path.join(dir_path, f"color_{idx}.png"))
    # results = det(img, stream=False)
    # show_glass_bbox(img, results, cup_idx)

    # cv2.imwrite(os.path.join(dir_path + '/annotated_glass_bb', f"color_{idx}_annotated.png"), img)
    keypoints, scores = wholebody(img)
    if not os.path.exists(os.path.join(dir_path + '/annotated', f"color_{idx}_annotated.png")):
        img = draw_skeleton(img, keypoints[0:1], scores, kpt_thr=0.1)
        cv2.imwrite(os.path.join(dir_path + '/annotated', f"color_{idx}_annotated.png"), img)
    keypoints_mat[i, :, :2] = keypoints[0]
    keypoints_mat[i, :, 2] = scores[0]
    keypoints_mat[i, :, 3] = idx
    # plt.imshow('img', img)
    # cv2.waitKey(1)

# np.save(f"{dir_path}/annotated/keypoints", keypoints_mat)
# test = np.load("/mnt/c/Users/Usager/Documents/Amedeo/rgbd_ms_cime/videos/img_0/annotated/keypoints.npy")
    # img = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)
    # cv2.imwrite(img_path.replace("img_0/", "img_0/annotated/"), img)
    # plt.imshow('img', img)
    # cv2.waitKey(1)