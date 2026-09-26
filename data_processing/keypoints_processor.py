import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from .camera_converter import CameraConverter
from .model_type import WholeBody
import numpy as np
import os
import cv2
import pickle
import open3d as o3d

try:
    from ultralytics import YOLO
except ImportError:
    pass
from .projection_utils import get_pc, perform_icp, transform_spheres
from .io_utils import write_trc, return_unique_keypoints
from .trajectory_utils import smooth_trajectory


class Keypoints3DProcessor:
    def __init__(self, yolo_model_path=None):
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None
        self.wholebody = WholeBody()
        self.cup_crops = None
        self.side = None
        self.tmp_3d_keypoints = None
        self.shoulder_weight = {"left": [0.5, 1], "right": [1, 1.5]}

    def _get_yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO("yolo26s.pt", verbose=False)
        return self._yolo_model

    def initialize_data(self, data_path, img_dir, camera, show_pc=False):
        if isinstance(camera, str):
            camera_config_path = camera
            camera = CameraConverter()
            camera.set_intrinsics(os.path.join(img_dir, camera_config_path))
            camera.set_extrinsics(os.path.join(img_dir, camera_config_path))
        self.data_path = data_path
        keypoints, idxs = return_unique_keypoints(data_path)
        self.idxs = idxs
        self.color_img_path = [os.path.join(img_dir, f"color_{int(i)}.png") for i in idxs]
        self.depth_img_path = [os.path.join(img_dir, f"depth_{int(i)}.png") for i in idxs]
        self.camera = camera
        self.keypoints = self.wholebody.get_minimal_keypoints(keypoints, 0).astype(int)
        self.keypoints_names = self.wholebody.minimal_set
        self._init_img()
        self._check_side()
        self._compute_ratio()
        self._prepare_thorax_icp(show_pc=show_pc)

    def _compute_ratio(self):
        left_arm = self.keypoints[0][
            [
                self.wholebody.get_index("left_shoulder"),
                self.wholebody.get_index("left_elbow"),
            ],
            :,
        ]
        right_arm = self.keypoints[0][
            [
                self.wholebody.get_index("right_shoulder"),
                self.wholebody.get_index("right_elbow"),
            ],
            :,
        ]
        len_left_arm = np.linalg.norm(left_arm[1] - left_arm[0])
        len_right_arm = np.linalg.norm(right_arm[1] - right_arm[0])
        self.shoulder_weight = {
            "left": [float(len_left_arm / len_right_arm), 1],
            "right": [1, float(len_right_arm / len_left_arm)],
        }

    def _check_side(self):
        keypoints_3d = self._get_3d_keypoints(self.keypoints[0], self.init_depth, idx=0, neighbourhood=15)
        left_points_idxs = [
            self.wholebody.get_index(name)
            for name in self.wholebody.minimal_set
            if name in ["left_shoulder", "left_elbow", "left_wrist"]
        ]
        right_points_idxs = [
            self.wholebody.get_index(name)
            for name in self.wholebody.minimal_set
            if name in ["right_shoulder", "right_elbow", "right_wrist"]
        ]
        mean_left = np.nanmean(keypoints_3d[left_points_idxs, -1])
        mean_right = np.nanmean(keypoints_3d[right_points_idxs, -1])
        if mean_left < mean_right:
            self.side = "left"
        else:
            self.side = "right"
        print(f"Detected side: {self.side} upper-limb")

    def _init_img(self):
        self.init_depth = cv2.imread(self.depth_img_path[0], cv2.IMREAD_ANYDEPTH)
        self.init_color = cv2.imread(self.color_img_path[0])

    def _get_3d_keypoints(self, points, depth_img, idx=0, neighbourhood=5, in_pixel=True):
        if self.tmp_3d_keypoints is not None and self.tmp_3d_keypoints[0] == idx:
            return self.tmp_3d_keypoints[1]
        keypoints_3d = self.camera.get_markers_pos_3d(
            points, depth_img, in_pixel=in_pixel, neighbourhood=neighbourhood, depth_in_meter=True
        )
        self.tmp_3d_keypoints = (idx, keypoints_3d)
        return keypoints_3d

    def _detect_cup(self, color_img):
        model = self._get_yolo_model()
        results = model(color_img)
        cup_boxes = []
        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls)] == "cup":
                    cup_boxes.append(box.xyxy[0].cpu().numpy())
        return cup_boxes

    def compute_3d_coordinates(self, track_thorax=True, track_cup=False):
        total_frames = self.idxs[-1] - self.idxs[0] + 1
        key_points_mat = np.full((total_frames, self.keypoints.shape[1] + len(self.thorax_spheres), 3), np.nan)
        cup_points_mat = np.full((total_frames, 3), np.nan)
        count = 0
        thorax_spheres = self.thorax_spheres
        color_img = None
        T_ref_current = np.eye(4)
        T_increment = np.eye(4)
        pc_thorax_keyframe = None
        count = 0
        for i, idx in enumerate(range(self.idxs[0], self.idxs[-1] + 1)):
            if idx not in self.idxs:
                continue
            points, img, color = self.keypoints[count], self.depth_img_path[count], self.color_img_path[count]
            count += 1

            depth_img = cv2.imread(img, cv2.IMREAD_ANYDEPTH)
            if track_cup:
                color_img = cv2.imread(color)
                # cup_bboxes = self._detect_cup(color_img)
                # if len(cup_bboxes) > 0:
                #     cup_center = ((cup_bboxes[0][0] + cup_bboxes[0][2]) // 2, (cup_bboxes[0][1] + cup_bboxes[0][3]) // 2)
                #     cup_depth = self.camera.get_depth_from_pixels(np.array(cup_center).astype(int), depth_img) * self.camera.depth_scale + 0.05
                #     center_meters = self.camera.get_markers_pos_in_meter([np.hstack([np.array(cup_center).astype(int), cup_depth])])
                # else:
                #     center_meters = np.array([np.nan, np.nan, np.nan])
                # pc_tmp = pc_from_rgbd(depth_img, color_img, self.camera)
                # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.080)
                # sphere.translate(center_meters)
                # sphere.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([pc_tmp, sphere])
                # cv2.rectangle(color_img, cup_bboxes[0][:2].astype(int), cup_bboxes[0][2:].astype(int), (0, 255, 0), 2)
                # cv2.circle(color_img, cup_center, 5, (255, 0, 0), -1)
                # cv2.imshow("cup detection", color_img)
                # cv2.waitKey(0)
            if track_thorax:
                pc_from_rgb = get_pc(depth_img, self.camera, color_img)
                pc_thorax = pc_from_rgb.crop(self.thorax_bbox)

                if pc_thorax.is_empty():
                    raise ValueError("Thorax point cloud is empty. Check the thorax crops and limit depth.")
                if i > 0:
                    T_ref_current = perform_icp(
                        ref_pc=pc_thorax_keyframe,
                        target_pc=pc_thorax,
                        threshold=0.01,
                        initial_guess=T_ref_current,
                        show=False,
                    )
                    thorax_spheres = transform_spheres(self.thorax_spheres, T_ref_current)
                    # T_increment = perform_icp(
                    #     pc_thorax_prev,
                    #     pc_thorax,
                    #     threshold=0.01,
                    #     initial_guess=T_increment,
                    #     show=False,
                    # )
                    # T_ref_current = T_increment @ T_ref_current
                    # thorax_spheres = transform_spheres(self.thorax_spheres, T_ref_current)
                    # o3d.visualization.draw_geometries([pc_thorax.rotate(T_ref_current, center=(0, 0, 0)), pc_thorax_prev])
                else:
                    pc_thorax_keyframe = o3d.geometry.PointCloud(pc_thorax)
            if track_cup and self.cup_crops is not None:
                raise NotImplementedError("Cup tracking is not implemented yet")
                # pc_cup = crop_pc(pc_tmp, self.cup_crops, self.cup_limit_depth, self.camera)
                # pc_cup = get_reduced_pc(depth_img, color_img, self.cup_crops, self.cup_limit_depth, self.camera)
                # if pc_cup.is_empty():

                # if count > 0:
                #     cup_sphere, cup_transformation = perform_icp(
                #         pc_cup_prev, pc_cup, cup_sphere, threshold=0.01, initial_guess=np.eye(4), show=False
                #     )
                #     self._move_crops_cup(cup_sphere)
                # pc_cup_prev = o3d.geometry.PointCloud(pc_cup)
            keypoints_3d = self.camera.get_markers_pos_3d(
                points, depth_img, in_pixel=False, neighbourhood=5, depth_in_meter=True
            )
            cup_points_mat[i, :] = np.array([np.nan, np.nan, np.nan])
            key_points_mat[i, :-len(self.thorax_spheres), :] = keypoints_3d.T
            key_points_mat[i, -len(self.thorax_spheres):, :] = np.stack(
                [thorax_spheres[j].get_center() for j in range(len(thorax_spheres))], axis=0
            )
        self.keypoints_3d = key_points_mat.copy()
        self.cup_points = cup_points_mat.copy()
        self.keypoints_names = self.wholebody.minimal_set + ['ster'] + [f"virtual_marker_{j}" for j in range(4)]
        return key_points_mat, cup_points_mat

    def _prepare_thorax_icp(self, shoulder_weight=None, show_pc=False):
        shoulder_weight = shoulder_weight if shoulder_weight is not None else self.shoulder_weight[self.side]
        pc = get_pc(self.init_depth, self.camera, self.init_color)
        points_3d = self.camera.get_markers_pos_3d(
            self.keypoints[0], self.init_depth, in_pixel=False, neighbourhood=5, depth_in_meter=True
        )

        ###### Work in the coordinate system where z is vertical
        points_vert = self.camera.align_with_z(points_3d.T)
        pc_copy = o3d.geometry.PointCloud(pc)
        pc_rot = pc_copy.rotate(self.camera.accel_rotation, center=(0, 0, 0))

        shoulder_pos = points_vert[
            [
                self.wholebody.get_index("right_shoulder"),
                self.wholebody.get_index("left_shoulder"),
            ],
            :,
        ]
        shoulder_pos_w = np.zeros_like(shoulder_pos)
        shoulder_pos_w[0, :] = shoulder_pos[0, :] * shoulder_weight[0]
        shoulder_pos_w[1, :] = shoulder_pos[1, :] * shoulder_weight[1]
        midpoint = np.sum(shoulder_pos_w, axis=0) / np.sum(shoulder_weight)
        dist2 = np.sum((np.array(pc_rot.points) - midpoint) ** 2, axis=1)
        idx = np.argmin(dist2)
        closest_point = np.array(pc_rot.points)[idx]
        closest_point[2] = closest_point[2] + 0.02
        close_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
        close_sphere.translate(closest_point)
        close_sphere.paint_uniform_color([0, 1, 0])
        self.sternum_sphere = close_sphere

        dist_should = np.linalg.norm(shoulder_pos[1] - shoulder_pos[0])
        mid_proj = midpoint + np.array([0, 0, 0.10])
        mid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.010).translate(mid_proj)
        mid_sphere.paint_uniform_color([1, 0, 0])

        x = shoulder_pos[1] - shoulder_pos[0]
        x = x / np.linalg.norm(x)

        z = np.array([0, 0, -1]) 

        # Make X exactly perpendicular to Z
        x = x - np.dot(x, z) * z
        x = x / np.linalg.norm(x)

        # Complete right-handed coordinate system
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)

        # Recompute X to guarantee perfect orthogonality
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)

        coordinate_frame = np.stack([x, y, z], axis=1)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=closest_point)
        mesh.rotate(coordinate_frame, center=closest_point)
        
        square_size = 0.1
        dx = [-square_size / 2, square_size / 2, -square_size / 2, square_size / 2]
        dz = [0, 0, -square_size, -square_size]
        self.thorax_spheres = [self.sternum_sphere]
        for dx_tmp, dz_tmp in zip(dx, dz):
            point = closest_point + dx_tmp * x + dz_tmp * z
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
            sphere.translate(point)
            sphere.paint_uniform_color([0, 1, 0])
            self.thorax_spheres.append(sphere)

        R = np.column_stack((x, y, z))
        origin = closest_point
        sx = dist_should * 0.8
        sy = 0.2
        sz = 0.6

        local_corners = np.array([
            [-sx/2, -sy/2, 0.05],
            [ sx/2, -sy/2, 0.05],
            [ sx/2,  sy/2, 0.05],
            [-sx/2,  sy/2, 0.05],
            [-sx/2, -sy/2,  -sz/2],
            [ sx/2, -sy/2,  -sz/2],
            [ sx/2,  sy/2,  -sz/2],
            [-sx/2,  sy/2,  -sz/2],
        ])

        global_corners = origin + local_corners @ R.T
        self.thorax_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(global_corners))

        ##### Translate everything in the camera coordinate system
        self.thorax_bbox.rotate(self.camera.accel_rotation.T, center=(0, 0, 0))
        self.thorax_spheres = [sphere.rotate(self.camera.accel_rotation.T, center=(0, 0, 0)) for sphere in self.thorax_spheres]
        if show_pc:
            o3d.visualization.draw_geometries([pc] + self.thorax_spheres + [self.thorax_bbox])

    def _prepare_cup_icp(self):
        detect_cup_boxes = self._detect_cup(self.init_color)
        if len(detect_cup_boxes) == 0:
            print("No cup detected")
            self.cup_crops, self.cup_limit_depth, self.cup_spheres = None, None, None
            return

        p1 = (int(detect_cup_boxes[0][0]) - 10, int(detect_cup_boxes[0][1]) - 10)
        p3 = (int(detect_cup_boxes[0][2]) + 10, int(detect_cup_boxes[0][3]) - 10)
        p2 = (int(detect_cup_boxes[0][0]) - 10, int(detect_cup_boxes[0][3]) - 10)
        p4 = (int(detect_cup_boxes[0][2]) + 10, int(detect_cup_boxes[0][1]) - 10)
        center = ((p1[0] + p3[0]) // 2, ((p1[1] + p3[1]) // 2) + 10)
        spheres = []
        for m, marker in enumerate([center, p1, p2, p3, p4]):
            x, y = np.array(marker).astype(int)
            cx, cy = self.camera.depth.ppx, self.camera.depth.ppy
            fx, fy = self.camera.depth.fx, self.camera.depth.fy
            if m == 0:
                z = self.init_depth[y, x]
                limit_depth = [
                    z - (0.08 * (1 / self.camera.depth_scale)),
                    z + (0.08 * (1 / self.camera.depth_scale)),
                ]
                limit_depth = np.clip(limit_depth, a_min=0, a_max=np.inf)
                Z = z * self.camera.depth_scale

            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            mid_point_3d = np.array([X, Y, Z])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.050)
            sphere.translate(mid_point_3d)
            sphere.paint_uniform_color([1, 0, 0])
            spheres.append(sphere)
        self.cup_crops, self.cup_limit_depth, self.cup_spheres = np.array([p1, p2, p3, p4]), limit_depth, spheres

    def _move_crops_cup(self, new_spheres):
        crop_pos = []
        for sphere in new_spheres[1:]:
            x, y = self.camera.get_marker_pos_in_pixel(sphere.get_center().reshape(1, 3)).squeeze()
            crop_pos.append((x, y))
        self.cup_crops = np.array(crop_pos)
        self.cup_limit_depth = new_spheres[0].get_center()[2] + np.array([-0.08, 0.08])
        self.cup_limit_depth /= self.camera.depth_scale

    @staticmethod
    def _plot_curve(title, idx_to_plot, plot_label, colors, data, input_data, output_dir):
        for s, side in enumerate(["left", "right"]):
            plt.figure(title + f" {side}")
            for i in range(len(idx_to_plot[s])):
                plt.plot(data[:, idx_to_plot[s][i], -1], label=plot_label[i], color=colors[i])
                plt.plot(input_data[:, idx_to_plot[s][i], -1], alpha=0.5, color=colors[i])
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{'_'.join(title.lower().split(' '))}_{side}.png"))
            plt.close()

    def _get_smoother_params(
        self, smoother_process_noise=None, smoother_measurement_noise=None, smoother_nis_threshold=None
    ):
        list_side = ["left", "right"]
        if isinstance(smoother_process_noise, dict) and "opposite_arm" in smoother_process_noise:
            smoother_process_noise = {
                self.side: smoother_process_noise["motion_arm"],
                list_side[np.where(np.array(list_side) != self.side)[0][0]]: smoother_process_noise["opposite_arm"],
                "center": smoother_process_noise["center"],
            }
        if isinstance(smoother_measurement_noise, dict) and "opposite_arm" in smoother_measurement_noise:
            smoother_measurement_noise = {
                self.side: smoother_measurement_noise["motion_arm"],
                list_side[np.where(np.array(list_side) != self.side)[0][0]]: smoother_measurement_noise["opposite_arm"],
                "center": smoother_measurement_noise["center"],
            }
        if isinstance(smoother_nis_threshold, dict) and "opposite_arm" in smoother_nis_threshold:
            smoother_nis_threshold = {
                self.side: smoother_nis_threshold["motion_arm"],
                list_side[np.where(np.array(list_side) != self.side)[0][0]]: smoother_nis_threshold["opposite_arm"],
                "center": smoother_nis_threshold["center"],
            }

        process_noise = (
            {self.side: 10, list_side[np.where(np.array(list_side) != self.side)[0][0]]: 1, "center": 10}
            if smoother_process_noise is None
            else smoother_process_noise
        )
        measurement_noise = (
            {
                self.side: 1e-3,
                list_side[np.where(np.array(list_side) != self.side)[0][0]]: 1e-4,
                "center": 1e-2,
            }
            if smoother_measurement_noise is None
            else smoother_measurement_noise
        )
        nis_threshold = (
            {
                self.side: 40.34,
                list_side[np.where(np.array(list_side) != self.side)[0][0]]: 10.34,
                "center": 5,
            }
            if smoother_nis_threshold is None
            else smoother_nis_threshold
        )
        return process_noise, measurement_noise, nis_threshold

    def post_process(
        self,
        apply_smoother=True,
        smoother_process_noise=None,
        smoother_measurement_noise=None,
        smoother_nis_threshold=None,
        align_with_z=True,
        save_plot=False,
        shoulder_weight=None,
        output_dir="",
    ):
        shoulder_weight = shoulder_weight if shoulder_weight is not None else self.shoulder_weight[self.side]
        post_process_3d = self.keypoints_3d.copy()
        side = "right" if self.side == "left" else "left"
        side = self.side
        idx_to_plot = []
        for side in ["left", "right"]:
            name_to_plot = [
                'nose',
                f"{side}_shoulder",
                f"{side}_elbow",
                f"{side}_wrist",
                f"{side}_hand_index1",
                f"{side}_hand_pinky1",
            ]
            plot_label = [name.split("_")[-1] for name in name_to_plot]
            colors = matplotlib.cm.tab10.colors[: len(name_to_plot)]
            idx_to_plot.append([self.wholebody.get_index(name) for name in name_to_plot])

        self.filtering = {
            "apply_smoother": apply_smoother,
            "smoother_process_noise": smoother_process_noise,
            "smoother_measurement_noise": smoother_measurement_noise,
            "smoother_nis_threshold": smoother_nis_threshold,
            "align_with_z": align_with_z,
        }

        input = post_process_3d.copy()
        if apply_smoother:
            process_noise, measurement_noise, nis_threshold = self._get_smoother_params(
                smoother_process_noise, smoother_measurement_noise, smoother_nis_threshold
            )
            post_process_3d = smooth_trajectory(
                input,
                dt=1 / self.camera.color.fps,
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                nis_threshold=nis_threshold,
                markers_name=self.keypoints_names,
            )
            if save_plot:
                self._plot_curve("After smoothing", idx_to_plot, plot_label, colors, post_process_3d, input, output_dir)

        if align_with_z:
            post_process_3d = self.camera.align_with_z(post_process_3d)
        self.post_process_3d = post_process_3d
        return post_process_3d

    def to_dict(self):
        dic_to_save = {
            "keypoints_3d": self.post_process_3d,
            "keypoints_2d": self.keypoints,
            "idxs": self.idxs,
            "depth_image_path": self.depth_img_path,
            "color_image_path": self.color_img_path,
            "key_points_idxs": self.wholebody.minimal_idxs,
            "key_points_names": self.keypoints_names,
            "filtering": self.filtering,
            "camera": self.camera.conf_data_dic,
            "side": self.side,
            # "cup_points": self.cup_points,
        }
        return dic_to_save

    def save(self, export_trc=True):
        dic_to_save = self.to_dict()
        with open(self.data_path.replace("keypoints.npy", "keypoints_3d.pkl"), "wb") as f:
            pickle.dump(dic_to_save, f)
        if export_trc:
            write_trc(
                self.post_process_3d.T,
                self.keypoints_names,
                self.data_path.replace("keypoints.npy", "keypoints_3d.trc"),
                self.camera.color.fps,
            )
        return True
