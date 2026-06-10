#!/usr/bin/env python3
"""A minimal example of streaming frames live from an Intel RealSense depth sensor."""
from __future__ import annotations

import json
import pyorerun
import rerun as rr
import argparse
import numpy as np

from camera_converter import CameraConverter
import os
import glob
import cv2
import csv


def get_3d_coordinates(keypoints, depth_img, camera):
    keypoints_3d = []
    for i in range(keypoints.shape[0]):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        z = depth_img[min(int(y), depth_img.shape[0] -1), min(int(x), depth_img.shape[1] -1)] * camera.depth_scale
        keypoints_3d.append([x, y, z])
    return camera.get_markers_pos_in_meter(np.array(keypoints_3d)).T

def run_realsense(num_frames: int | None, trial=None, part=None) -> None:
    # Visualize the data as RDF
    camera_conf_file = f"camera_config.json"
    # rgbd = RgbdImages(path_to_camera_config_file)
    # display_option = pyorerun.DisplayModelOptions()

    # markers_names = marker_data["key_point_reduced_names"]
    # connections = marker_data["key_points_connections"]
    # markers_3d = marker_data["key_points_filtered"]
    # mot_file = fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons\ik_mot.mot"
    # q = pyorerun.OsimTimeSeries(mot_file, model_path).q_in_radian


    # rr.init("world", spawn=True)
    rr.log("", rr.ViewCoordinates.RDF, static=True
        # timeless=True,
        )
    # model.model.options.transparent_mesh = False
    converter = CameraConverter()
    converter.set_intrinsics(camera_conf_file)
    converter.set_extrinsics(camera_conf_file)
    depth_intr = converter.depth
    rgb_intr = converter.depth
    converter.depth_to_color = np.eye(4)
    trans = converter.depth_to_color[:3, 3]
    rot = converter.depth_to_color[:3, :3]

    rr.log(
        "depth/image",
        rr.Pinhole(
            resolution=[depth_intr.width, depth_intr.height],
            focal_length=[depth_intr.fx, depth_intr.fy],
            principal_point=[depth_intr.ppx, depth_intr.ppy],
        ),
        static=True,
    )

    rr.log(
        "rgb",
        rr.Transform3D(
            translation=trans,
            mat3x3=rot,
            from_parent=True,
        ),
        static=True,
    )

    rr.log(
        "rgb/image",
        rr.Pinhole(
            resolution=[rgb_intr.width, rgb_intr.height],
            focal_length=[rgb_intr.fx, rgb_intr.fy],
            principal_point=[rgb_intr.ppx, rgb_intr.ppy],
        ),
        static=True,

    )
    dir_path = r"C:\Users\Usager\Documents\Amedeo\rgbd_ms_cime\videos\img_0"
    all_image = os.listdir(dir_path)
    all_image = [os.path.join(dir_path, im) for im in all_image if im.endswith(".png")]
    idxs = [int(os.path.basename(file).split("_")[1].split(".")[0]) for file in all_image]
    idxs = sorted(idxs)
    keypoints_mat = np.load(os.path.join(dir_path, "annotated", "keypoints.npy"))
    camera = CameraConverter()
    camera.set_intrinsics("camera_config.json")
    camera.set_extrinsics("camera_config.json")
    # rr.log(
    #     "/",
    #     rr.AnnotationContext(
    #         rr.ClassDescription(
    #             info=rr.AnnotationInfo(id=1, label="Person"),
    #             keypoint_annotations=[rr.AnnotationInfo(id=i, label=str(i)) for i in range(25)],
    #             keypoint_connections=connections,
    #         )
    #     ),
    #     static=True,
    # )
    frame_nr = 0
    while True:
        if frame_nr >= len(idxs):
            break
        rr.set_time_sequence("frame_nr", frame_nr)
        depth_image = cv2.imread(os.path.join(dir_path, f"depth_{idxs[frame_nr]}.png"), cv2.IMREAD_ANYDEPTH)
        color_image = cv2.cvtColor(
            cv2.imread(os.path.join(dir_path, f"color_{idxs[frame_nr]}.png")), cv2.COLOR_BGR2RGB
        )
        if depth_image is None or color_image is None:
            frame_nr += 1
            continue

        keypoints_2d = keypoints_mat[frame_nr][keypoints_mat[frame_nr][..., -1] > 0.6][:, :2]
        keypoints_3d = get_3d_coordinates(keypoints_2d, depth_image, camera)
        depth_image = np.where(
            (depth_image > 2 / converter.depth_scale) | (depth_image <= 0.3 / converter.depth_scale),
            0,
            depth_image,
        )

        # q, _, _ = msk.compute_inverse_kinematics(keypoints_3d[:, :, None],
        #     method=InverseKinematicsMethods.BiorbdLeastSquare,
        # )
        # except:
        #     print(f"frame {idxs[frame_nr]} not found")
        #     continue
        rr.log("depth/image", rr.DepthImage(depth_image, meter=1 / converter.depth_scale))
        rr.log("rgb/image", rr.Image(color_image))
        rr.log("rgb/image/2d_keypoints", rr.Points2D(keypoints_2d[:, :2],
                                                           # colors=(0, 125, 255),
                                                           radii=4,
                                                           keypoint_ids=list(range(keypoints_2d.shape[0])),
                                                             class_ids=1, show_labels=False))
        rr.log("keypoints", rr.Points3D(keypoints_3d, colors=(0, 125, 255), radii=0.01,
                                              keypoint_ids=list(range(keypoints_2d.shape[0])),
                                                class_ids=1, show_labels=False))
        # rr.log("world/keypoints markers", rr.Points3D(markers_model[:, :, frame_nr].T, colors=(125, 0, 255),
        #                                       radii=0.01,
        #                                       ))
        # phase_rerun.update_animated_model(q[:, frame_nr])
        # model.to_rerun(q[:, frame_nr])
        frame_nr += 1
    print(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streams frames from a connected realsense depth sensor.")
    parser.add_argument("--num-frames", type=int, default=None, help="The number of frames to log")
    #
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "rerun_example_live_depth_sensor")

    run_realsense(
        args.num_frames,
        "gear_10", "P11")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
