#!/usr/bin/env python3
"""A minimal example of streaming frames live from an Intel RealSense depth sensor."""

import json
import pyorerun
import rerun as rr
import argparse
import numpy as np

from save_load import load
from camera_converter import CameraConverter
import os
import glob
import cv2
import pickle
import csv

def q_from_mot(trc_file):
    rows = []
    with open(trc_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r, row in enumerate(reader):
            if r >= 11:
                rows.append([float(x) for x in row[1:]])
            if r == 10:
                headers = row[1:]
    mot = np.array(rows).T
    q = np.zeros_like(mot)
    q[:3, :] = mot[3:6, :]
    q[3:, :] = np.radians(np.concatenate([mot[:3, :], mot[6:, :]], axis=0))
    return q


def run_realsense(num_frames: int | None, trial=None, part=None) -> None:
    # Visualize the data as RDF
    data_dir = r'D:\Documents\Programmation\markerless_drinking_task\videos\20260514_112323\annotated'
    camera_conf_file = r"D:\Documents\Programmation\markerless_drinking_task\camera_config.json"
    model_path = r"D:\Documents\Programmation\markerless_drinking_task\opensim\wu_modified_markerless.osim"
    # rgbd = RgbdImages(path_to_camera_config_file)
    display_option = pyorerun.DisplayModelOptions()
    display_option.mesh_path = r"D:\Documents\Programmation\markerless_drinking_task\opensim\Geometry"
    model = pyorerun.ModelUpdater.from_file(model_path, options=display_option)
    with open(os.path.join(data_dir, "keypoints_3d.pkl"), "rb") as f:
        marker_dic = pickle.load(f)
    # marker_data = load(markers_file)
    markers_names = marker_dic["key_points_names"]
    # connections = marker_dic["key_points_connections"]
    markers_3d = marker_dic["keypoints_3d"]
    marker_2d = marker_dic["keypoints_2d"]
    img_path = marker_dic["depth_image_path"]
    idxs_img = [int(id) for i, id in enumerate(marker_dic["idxs"])]
    mot_file = os.path.join(data_dir, "output_motion.mot")
    q = pyorerun.OsimTimeSeries(mot_file, model_path).q_in_radian


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
    # display_option = pyorerun.DisplayModelOptions()
    display_option.mesh_color = (77, 77, 255)
    while True:
        # if frame_nr >= q.shape[1] - 1:
            # break
        if frame_nr >= len(idxs_img):
            break
        rr.set_time_sequence("frame_nr", frame_nr)
        depth_image = cv2.imread(img_path + f"\depth_{idxs_img[frame_nr]}.png", cv2.IMREAD_ANYDEPTH)
        depth_image = np.where(
            (depth_image > 2 / converter.depth_scale) | (depth_image <= 0.5 / converter.depth_scale),
            0,
            depth_image,
        )
        color_image = cv2.cvtColor(
            cv2.imread(img_path + f"\color_{idxs_img[frame_nr]}.png"), cv2.COLOR_BGR2RGB
        )
        # q, _, _ = msk.compute_inverse_kinematics(keypoints_3d[:, :, None],
        #     method=InverseKinematicsMethods.BiorbdLeastSquare,
        # )
        # except:
        #     print(f"frame {idxs[frame_nr]} not found")
        #     continue
        rr.log("depth/image", rr.DepthImage(depth_image, meter=1 / converter.depth_scale))
        rr.log("rgb/image", rr.Image(color_image))
        rr.log("rgb/image/2d_keypoints", rr.Points2D(marker_2d[frame_nr],
                                                           # colors=(0, 125, 255),
                                                           radii=4,
                                                           keypoint_ids=list(range(marker_2d.shape[1])), class_ids=1, show_labels=False))
        rr.log("keypoints", rr.Points3D(markers_3d[frame_nr, :, :], colors=(0, 125, 255), radii=0.01,
                                              keypoint_ids=list(range(marker_2d.shape[1])),
                                                class_ids=1, show_labels=False))
        # rr.log("world/keypoints markers", rr.Points3D(markers_model[:, :, frame_nr].T, colors=(125, 0, 255),
        #                                       radii=0.01,
        #                                       ))
        # phase_rerun.update_animated_model(q[:, frame_nr])
        model.to_rerun(q[:, frame_nr])
        frame_nr += 1


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
