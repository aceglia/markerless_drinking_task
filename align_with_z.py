import os

import pyrealsense2 as rs
import numpy as np
import cv2
from camera_converter import CameraConverter

if __name__ == "__main__":

    dir_path = r'F:\CIME_MS\adnane\20260610_121314.bag'
    dir_config = r'F:\CIME_MS\adnane\20260610_121314'
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(dir_path, repeat_playback=False)

    profile = pipe.start(cfg)
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    accel_profile = profile.get_stream(rs.stream.accel).as_motion_stream_profile()
    extr = accel_profile.get_extrinsics_to(depth_profile)
    converter = CameraConverter(use_camera=True)
    converter.set_intrinsics(pipe)
    converter.set_extrinsics(pipe)
    # converter.set_extrinsics_to_accel(pipe)
    # Wait a few frames so everything is initialized
    g = []
    for _ in range(20):
        frames = pipe.wait_for_frames()
        converter.add_accel_frame(frames)


        # accel = frames.first_or_default(rs.stream.accel)
        # a = accel.as_motion_frame().get_motion_data()
        # g.append(np.array([a.x, a.y, a.z]))
    converter.save_config(os.path.join(dir_config, "camera_config.json"))
    g = np.mean(np.array(g), axis=0)
    g /= np.linalg.norm(g)

    from scipy.spatial.transform import Rotation

    target = np.array([0, 0, -1])      # Z upward convention

    axis = np.cross(g, target)
    axis_norm = np.linalg.norm(axis)

    if axis_norm > 1e-8:
        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(g, target), -1, 1))
        R = Rotation.from_rotvec(axis * angle).as_matrix()
    else:
        R = np.eye(3)