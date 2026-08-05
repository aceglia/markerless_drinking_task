import cv2
from data_processing.utils_onnrx import Wholebody
import pyrealsense2 as rs
from data_processing.camera_converter import CameraConverter

import os
os.environ["YOLO_VERBOSE"] = 'False'
import numpy as np

device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style (required for animals), False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend,
                       device=device)

dir_path = '/mnt/c/Users/Usager/Documents/Amedeo/rgbd_ms_cime/videos/tests_lexie'
dir_path = r"C:\Users\neuromobility_lab\Documents\CIME_MS\test_002"
all_bag_files = os.listdir(dir_path)
all_bag_files_names = [f.removesuffix(".bag") for f in all_bag_files if f.endswith("20260729_113412.bag")]

pipeline = rs.pipeline()

# Create a config object
for file in all_bag_files_names:
    print("Processing file: ", file)
    
    tmp_path = os.path.join(dir_path, file)

    os.makedirs(tmp_path, exist_ok=True)
    # if not os.path.exists(tmp_path + ".bag"):
    #     shutil.copy2(os.path.join(dir_path, file + ".bag"), tmp_path + ".bag")
    bag_path = tmp_path + ".bag"

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    profile = pipeline.start(config)
    converter = CameraConverter(use_camera=True)
    converter.set_intrinsics(pipeline)
    converter.set_extrinsics(pipeline)
    playback = profile.get_device().as_playback()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    accel_profile = profile.get_stream(rs.stream.accel).as_motion_stream_profile()
    playback.set_real_time(False)
    align_to = rs.stream.color
    align = rs.align(align_to)
    keypoints_mat = None
    os.makedirs(tmp_path + '/annotated', exist_ok=True)
    import time 
    tic = time.time()
    count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(1500)
            frames = align.process(frames)
            if count > 50:
                converter.save_config(os.path.join(tmp_path, "camera_config.json"))
            else:
                converter.add_accel_frame(frames)
                count += 1
            frame_number = frames.frame_number
            # print("processing frame : ", frame_number)
            # Get depth frame
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_raw = np.asanyarray(depth_frame.get_data())
            color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
            if not os.path.exists(os.path.join(tmp_path, f"color_{frame_number}.png")):
                cv2.imwrite(os.path.join(tmp_path, f"color_{frame_number}.png"), color_image)
                cv2.imwrite(os.path.join(tmp_path, f"depth_{frame_number}.png"), depth_raw)
            img = color_image
            keypoints, scores = wholebody(img)
            center = img.shape[0] // 2, img.shape[1] // 2
            mean_keypoints = keypoints.mean(axis=1)
            distance = np.linalg.norm(mean_keypoints - center, axis=1)
            min_dist = np.argmin(distance)
            if not os.path.exists(os.path.join(tmp_path + '/annotated', f"color_{frame_number}_annotated.png")):
                img = wholebody.draw_skeleton(img, keypoints[min_dist:min_dist + 1], scores, kpt_thr=0.5)
                cv2.imwrite(os.path.join(tmp_path + '/annotated', f"color_{frame_number}_annotated.png"), img)
            idx = np.zeros_like(scores) + frame_number
            global_mat = np.concatenate((keypoints[0], scores[0][:, None], idx[0][:, None]), axis = -1)
            keypoints_mat = np.vstack([keypoints_mat, global_mat[None]]) if keypoints_mat is not None else global_mat[None]
    except:
        if count <= 50:
            converter.save_config(os.path.join(tmp_path, "camera_config.json"))
        pass
    finally:
        if count <= 50:
            converter.save_config(os.path.join(tmp_path, "camera_config.json"))
        print(time.time()-tic)
        pipeline.stop()
        np.save(tmp_path + '/annotated/keypoints', keypoints_mat)