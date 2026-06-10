import cv2
from rtmlib import Wholebody, draw_skeleton, Body
import os
import time
import numpy as np
# import matplotlib.pyplot as plt


device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style (required for animals), False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend,
                       device=device)
# dir_path = "/mnt/c/Users/Usager/Documents/Amedeo/rgbd_ms_cime/videos/img_0"
vid_path = '/mnt/c/Users/Usager/Documents/Amedeo/rgbd_ms_cime/videos/test_CIME_LOC'
all_vid_fold = os.listdir(vid_path)
n_to_process = len(all_vid_fold)
for f, fold in enumerate(all_vid_fold):
    video = os.path.join(vid_path, fold, os.listdir(os.path.join(vid_path, fold))[0])
    images_output_dir = os.path.join(os.path.dirname(video), 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(images_output_dir + '/annotated', exist_ok=True)
    if os.path.exists(images_output_dir + '/annotated' + '/keypoints_tmp.npy'):
        os.remove(images_output_dir + '/annotated' + '/keypoints_tmp.npy')
    if os.path.exists(images_output_dir + '/annotated' + '/keypoints.npy'):
        os.remove(images_output_dir + '/annotated' + '/keypoints.npy')

    print("processing video: ", video, f"{f}/{n_to_process}")
    cap = cv2.VideoCapture(video)
    cap.get
    frame_count = 0
    keypoints_mat = None
    tic = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not os.path.exists(os.path.join(images_output_dir, f'color_{frame_count}.png')):
            cv2.imwrite(os.path.join(images_output_dir, f'color_{frame_count}.png'), frame)
        keypoints, scores = wholebody(frame)
        img = draw_skeleton(frame, keypoints[0:1], scores, kpt_thr=0.4)
        if not os.path.exists(os.path.join(images_output_dir + '/annotated', f"color_{frame_count}_annotated.png")):
            cv2.imwrite(os.path.join(images_output_dir + '/annotated', f"color_{frame_count}_annotated.png"), img)
        global_mat = np.concatenate((keypoints[0], scores[0][:, None]), axis = -1)
        keypoints_mat = np.vstack([keypoints_mat, global_mat[None]]) if keypoints_mat is not None else global_mat[None]
        frame_count += 1

        if frame_count % 2000 == 0:
            np.save(images_output_dir + '/annotated' + '/keypoints_tmp', keypoints_mat)
            
    total_time = time.perf_counter() - tic
    print("Processing time:", np.round(total_time, 3), f"(time per frame: {np.round(total_time / frame_count, 3)})")
    np.save(images_output_dir + '/annotated' + '/keypoints', keypoints_mat)
    if os.path.exists(images_output_dir + '/annotated' + '/keypoints_tmp.npy'):
        os.remove(images_output_dir + '/annotated' + '/keypoints_tmp.npy')
    cap.release()
