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
from .trajectory_utils import remove_outliers, clean_trajectory, filter_points_3d


class Keypoints3DProcessor:
    def __init__(self, yolo_model_path=None, camera_frames_range=None):
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None
        self.wholebody = WholeBody()
        self.camera_frames_range = camera_frames_range
        self.cup_crops = None
        self.side = None
        self.tmp_3d_keypoints = None

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
        if self.camera_frames_range is not None:
            start_frame, end_frame = self.camera_frames_range
            idxs = idxs[(idxs >= start_frame) & (idxs <= end_frame)]
        self.idxs = idxs
        self.color_img_path = [os.path.join(img_dir, f"color_{int(i)}.png") for i in idxs]
        self.depth_img_path = [os.path.join(img_dir, f"depth_{int(i)}.png") for i in idxs]
        self.camera = camera
        self.keypoints = self.wholebody.get_minimal_keypoints(keypoints, 0).astype(int)
        self.keypoints_names = self.wholebody.minimal_set
        self._init_img()
        self._check_side()
        self._prepare_thorax_icp(show_pc=show_pc)
    
    def _check_side(self):
        keypoints_3d = self._get_3d_keypoints(self.keypoints[0], self.init_depth, idx=0, neighbourhood=5)
        left_points_idxs = [self.wholebody.get_index(name) for name in self.wholebody.minimal_set if name in ['left_shoulder', 'left_elbow', 'left_wrist']]
        right_points_idxs = [self.wholebody.get_index(name) for name in self.wholebody.minimal_set if name in ['right_shoulder', 'right_elbow', 'right_wrist']]
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
        keypoints_3d = self.camera.get_markers_pos_3d(points, depth_img, in_pixel=in_pixel, neighbourhood=neighbourhood, depth_in_meter=True)
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
        key_points_mat = np.empty((self.keypoints.shape[0], self.keypoints.shape[1] + 4, 3))
        cup_points_mat = np.empty((self.keypoints.shape[0], 3))
        count = 0
        thorax_spheres = self.thorax_spheres
        color_img = None
        T_ref_current = np.eye(4)
        for points, img, color in zip(self.keypoints, self.depth_img_path, self.color_img_path):
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
                # pc_thorax = crop_pc(pc_tmp, self.thorax_crops, self.thorax_limit_depth, self.camera)
                # pc_thorax = get_reduced_pc(depth_img, color_img, self.thorax_crops, self.thorax_limit_depth, self.camera)
                pc_from_rgb = get_pc(depth_img, self.camera, color_img)
                pc_thorax = pc_from_rgb.crop(self.thorax_bbox)

                if pc_thorax.is_empty():
                    raise ValueError("Thorax point cloud is empty. Check the thorax crops and limit depth.")
                if count > 0:
                    T_increment = perform_icp(
                        pc_thorax_prev,
                        pc_thorax,
                        threshold=0.01,
                        initial_guess=np.eye(4),
                        show=False,
                    )
                    T_ref_current = T_increment @ T_ref_current
                    thorax_spheres = transform_spheres(self.thorax_spheres, T_ref_current)
                pc_thorax_prev = o3d.geometry.PointCloud(pc_thorax)
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
            keypoints_3d = self.camera.get_markers_pos_3d(points, depth_img, in_pixel=False, neighbourhood=5, depth_in_meter=True)

            key_points_mat[count, :-4, :] = keypoints_3d.T
            key_points_mat[count, -4:, :] = np.stack([thorax_spheres[i].get_center() for i in range(len(thorax_spheres))], axis=0)
            count += 1
        self.keypoints_3d = key_points_mat.copy()
        self.cup_points = cup_points_mat.copy()
        self.keypoints_names = self.wholebody.minimal_set + [f"virtual_marker_{i}" for i in range(4)]
        return key_points_mat, cup_points_mat

    def _prepare_thorax_icp(self, show_pc=False):
        pc = pc_from_rgbd(self.init_depth, self.init_color, self.camera)
        points_3d = self.camera.get_markers_pos_3d(self.keypoints[0], self.init_depth, in_pixel=False, neighbourhood=5, depth_in_meter=True)
        points_vert = self.camera.align_with_z(points_3d.T)
        pc_copy = o3d.geometry.PointCloud(pc)
        pc_rot = pc_copy.rotate(self.camera.accel_rotation, center=(0, 0, 0))

        # coordinate system 
        co_rotate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        co_rotate = co_rotate.rotate(self.camera.accel_rotation, center=(0, 0, 0))
        shoulder_pos = points_vert[
            [self.wholebody.get_index("right_shoulder"), self.wholebody.get_index("left_shoulder"), 
             self.wholebody.get_index("right_elbow"), self.wholebody.get_index("left_elbow")], :
        ]

        midpoint = np.mean(shoulder_pos, axis=0)
        if self.side == "left":
            weight = [0.5, 1]
        else:
            weight = [1, 1.5]
        shoulder_pos = points_vert[
            [self.wholebody.get_index("right_shoulder"), self.wholebody.get_index("left_shoulder"), 
             ], :
        ]        
        midpoint = np.einsum('i,ij->j', weight, shoulder_pos) / np.sum(weight)
        dist_should = np.linalg.norm(shoulder_pos[1] - shoulder_pos[0])
        mid_proj = midpoint + np.array([0, 0, dist_should * 0.5])
        dist2 = np.sum((np.array(pc_rot.points) - mid_proj)**2, axis=1)
        idx = np.argmin(dist2)
        
        closest_point = np.array(pc_rot.points)[idx]
        radius = 0.2
        dist = np.linalg.norm(np.array(pc_rot.points) - closest_point, axis=1)
        mask = dist <= radius
        roi = np.array(pc_rot.points)[mask]

        pts = roi[::5]

        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        _, _, Vh = np.linalg.svd(pts_centered, full_matrices=False)
        first_axis = shoulder_pos[1] - shoulder_pos[0]
        normal = Vh[-1, :]
        n = normal / np.linalg.norm(normal)
        first_axis /= np.linalg.norm(first_axis)
        first_axis = first_axis - np.dot(first_axis, n) * n
        first_axis /= np.linalg.norm(first_axis)

        y = np.cross(n, first_axis)
        y /= np.linalg.norm(y)

        # create four points at the corners of a square centered at closest_point, with normal vector n and first_axis as one of the axes
        square_size = 0.1
        self.thorax_spheres = []
        for dx in [-square_size / 2, square_size / 2]:
            for dy in [-square_size / 2, square_size / 2]:
                point = closest_point + dx * first_axis + dy * y
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
                sphere.translate(point)
                sphere.paint_uniform_color([0, 1, 0])
                sphere.rotate(self.camera.accel_rotation.T, center=(0, 0, 0))
                self.thorax_spheres.append(sphere)

        coordinate_frame = np.stack([first_axis, y, normal], axis=1)
        self.thorax_bbox = o3d.geometry.OrientedBoundingBox(
            center=closest_point + [0, 0.05, 0.02],
            R=coordinate_frame,
            extent=np.array([dist_should * 0.8, dist_should, 0.16])  # x, y, z lengths
        )        
        self.thorax_bbox.rotate(self.camera.accel_rotation.T, center=(0, 0, 0))
        if show_pc:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=closest_point)
            axes.rotate(coordinate_frame, center=closest_point)
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

    def post_process(
        self,
        remove_outliers_on_diff=True,
        remove_outliers_on_sd=True,
        cluster_base_filter=True,
        lp_filter=True,
        lp_filter_cutoff=4,
        lp_filter_order=2,
        cluster_eps_list=[0.1, 0.08, 0.06, 0.04, 0.02, 0.01],
        idxs_for_clustering=None,
        align_with_z=True,
        plot=False,
    ):
        post_process_3d = self.keypoints_3d.copy()
        side = "right" if self.side == "left" else "left"
        side = self.side
        idx_to_plot = [self.wholebody.get_index(name) for name in [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]]
        idx_to_plot = [self.wholebody.get_index(name) for name in [f"{side}_hand_index1", f"{side}_hand_pinky1", f"{side}_wrist"]]

        self.filtering = {"remove_outliers_on_diff": remove_outliers_on_diff,
                           "cluster_base_filter": cluster_base_filter,
                             "lp_filter": lp_filter, 
                             "lp_filter_cutoff": lp_filter_cutoff,
                             "lp_filter_order": lp_filter_order,
                             "cluster_eps_list": cluster_eps_list,
                             "align_with_z": align_with_z}

        if cluster_base_filter:
            input = post_process_3d.copy()
            post_process_3d = clean_trajectory(post_process_3d, idxs=idxs_for_clustering, eps_list=cluster_eps_list, ratio_threshold=3)
            if plot:
                plt.figure('After removing on cluster')
                plt.plot(post_process_3d[:, idx_to_plot, 2])
                plt.plot(input[:, idx_to_plot, 2], alpha=0.1)
                
        if remove_outliers_on_sd:
            input = post_process_3d.copy()
            post_process_3d = remove_outliers(post_process_3d, on_diff=False)
            if plot:
                plt.figure('After removing outliers on sd')
                plt.plot(post_process_3d[:, idx_to_plot, 2])
                plt.plot(input[:, idx_to_plot, 2], alpha=0.1)
                
        if remove_outliers_on_diff:
            input = post_process_3d.copy()
            post_process_3d = remove_outliers(post_process_3d, on_diff=True)
            if plot:
                plt.figure('After removing outliers on diff')
                plt.plot(post_process_3d[:, idx_to_plot, 2])
                plt.plot(input[:, idx_to_plot, 2], alpha=0.1)

        if lp_filter:
            input = post_process_3d.copy()
            post_process_3d = filter_points_3d(
                post_process_3d, lp_filter_cutoff, self.camera.color.fps, order=lp_filter_order
            )
            if plot:
                plt.figure('After low-pass filtering')
                plt.plot(post_process_3d[:, idx_to_plot, 2])
                plt.plot(input[:, idx_to_plot, 2], alpha=0.1)
                plt.show(block=True)
                
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

    def save(self, export_trc=False):
        dic_to_save = self.to_dict()
        with open(self.data_path.replace("keypoints.npy", "keypoints_3d.pkl"), "wb") as f:
            pickle.dump(dic_to_save, f)
        if export_trc:
            write_trc(
                self.post_process_3d.T,
                self.wholebody.minimal_set + [f"virtual_marker_{i}" for i in range(4)],
                self.data_path.replace("keypoints.npy", "keypoints_3d.trc"),
                self.camera.color.fps,
            )
        return True

