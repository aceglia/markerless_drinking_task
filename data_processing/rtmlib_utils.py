import os

os.environ["YOLO_VERBOSE"] = "False"
import cv2
from .utils_onnrx import Wholebody
from .realsense_utils import init_db3, get_aligned_depth, get_frame

import numpy as np


def get_wholebody_model(device="cuda", backend="onnxruntime", mode="balanced", openpose_skeleton=False):
    wholebody = Wholebody(
        to_openpose=openpose_skeleton,
        mode=mode,  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
        backend=backend,
        device=device,
    )
    return wholebody


def get_image_range(dir_path):
    if os.path.exists(os.path.join(dir_path, "range.json")):
        import json

        with open(os.path.join(dir_path, "range.json"), "r") as f:
            data = json.load(f)
            start_frame = data["range"][0]
            end_frame = data["range"][1]
        if end_frame == -1:
            end_frame = np.inf
        return start_frame, end_frame
    else:
        return -1, np.inf


def process_bag_file(file, wholebody, create_anotated_images=True):
    print("Processing file: ", file)
    if os.path.exists(file):
        bag_path = file
    else:
        return
    start_frame, end_frame = get_image_range(file.removesuffix(".db3"))
    pipeline, align, converter, base_dir, tmp_path = init_db3(bag_path)
    keypoints_mat = None
    if create_anotated_images:
        os.makedirs(tmp_path + "/annotated", exist_ok=True)
    import time

    tic = time.time()
    count = 0
    try:
        while True:
            frames = get_frame(pipeline, align, timeout=1500)
            if count > 50:
                converter.save_config(os.path.join(base_dir, "camera_config.json"))
            else:
                converter.add_accel_frame(frames)
                count += 1
            frame_number = frames.frame_number

            if frame_number < start_frame or frame_number > end_frame:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_raw = get_aligned_depth(align, depth_raw)
            color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
            if not os.path.exists(os.path.join(tmp_path, f"color_{frame_number}.png")):
                cv2.imwrite(os.path.join(tmp_path, f"color_{frame_number}.png"), color_image)

            if not os.path.exists(os.path.join(tmp_path, f"depth_{frame_number}.png")):
                cv2.imwrite(os.path.join(tmp_path, f"depth_{frame_number}.png"), depth_raw)
            img = color_image
            keypoints, scores = wholebody(img)
            center = img.shape[0] // 2, img.shape[1] // 2
            mean_keypoints = keypoints.mean(axis=1)
            distance = np.linalg.norm(mean_keypoints - center, axis=1)
            min_dist = np.argmin(distance)
            if create_anotated_images:
                img = wholebody.draw_skeleton(img, keypoints[min_dist : min_dist + 1], scores, kpt_thr=0)
                cv2.imwrite(os.path.join(tmp_path + "/annotated", f"color_{frame_number}_annotated.png"), img)
            idx = np.zeros_like(scores) + frame_number
            global_mat = np.concatenate(
                (keypoints[min_dist], scores[min_dist][:, None], idx[min_dist][:, None]), axis=-1
            )
            keypoints_mat = (
                np.vstack([keypoints_mat, global_mat[None]]) if keypoints_mat is not None else global_mat[None]
            )
    except:
        if count <= 50:
            converter.save_config(os.path.join(base_dir, "camera_config.json"))
        pass
    finally:
        if count <= 50:
            converter.save_config(os.path.join(base_dir, "camera_config.json"))
        print(time.time() - tic)
        pipeline.stop()
        np.save(base_dir + "/results/keypoints", keypoints_mat)
    return base_dir, tmp_path
