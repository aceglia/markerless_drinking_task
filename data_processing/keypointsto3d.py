import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from camera_converter import CameraConverter
from model_type import WholeBody
import numpy as np
import os
import cv2
import pickle
import open3d as o3d
try:
    from ultralytics import YOLO
except ImportError:
    pass
from projection_utils import get_mid_point, get_reduced_pc, perform_icp, pc_from_rgbd, crop_pc
from io_utils import write_trc, return_unique_keypoints
from trajectory_utils import remove_outliers, clean_trajectory, filter_points_3d


class Keypoints3DProcessor:
    def __init__(self, yolo_model_path=None):
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None
        self.wholebody = WholeBody()

    def _get_yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO("yolo26s.pt", verbose=False)
        return self._yolo_model

    def initialize_data(self, data_path, img_dir, camera):
        if isinstance(camera, str):
            camera_config_path = camera
            camera = CameraConverter()
            camera.set_intrinsics(os.path.join(dir, camera_config_path))
            camera.set_extrinsics(os.path.join(dir, camera_config_path))
        self.data_path = data_path
        keypoints, idxs = return_unique_keypoints(data_path)
        self.idxs = idxs
        self.color_img_path = [os.path.join(img_dir, f"color_{int(i)}.png") for i in idxs]
        self.depth_img_path = [os.path.join(img_dir, f"depth_{int(i)}.png") for i in idxs]
        self.camera = camera
        self.keypoints = self.wholebody.get_minimal_keypoints(keypoints, 0)
        self.keypoints_names = self.wholebody.minimal_set
        self._init_img()
        self._prepare_thorax_icp()

    def _init_img(self):
        self.init_depth = cv2.imread(self.depth_img_path[0], cv2.IMREAD_ANYDEPTH)
        self.init_color = cv2.imread(self.color_img_path[0])

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
        for points, img, color in zip(self.keypoints, self.depth_img_path, self.color_img_path):
            depth_img = cv2.imread(img, cv2.IMREAD_ANYDEPTH)
            if track_thorax or track_cup:
                color_img = cv2.imread(color)
                cup_bboxes = self._detect_cup(color_img)
                if len(cup_bboxes) > 0:
                    cup_center = ((cup_bboxes[0][0] + cup_bboxes[0][2]) // 2, (cup_bboxes[0][1] + cup_bboxes[0][3]) // 2)
                    cup_depth = self.camera.get_depth_from_pixels(np.array(cup_center).astype(int), depth_img) * self.camera.depth_scale + 0.05
                    center_meters = self.camera.get_markers_pos_in_meter([np.hstack([np.array(cup_center).astype(int), cup_depth])])
                else:
                    center_meters = np.array([np.nan, np.nan, np.nan])
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
                pc_thorax = get_reduced_pc(depth_img, color_img, self.thorax_crops, self.thorax_limit_depth, self.camera)
                if count > 0:
                    thorax_spheres, thorax_transformation = perform_icp(
                        pc_thorax_prev,
                        pc_thorax,
                        thorax_spheres,
                        threshold=0.01,
                        initial_guess=np.eye(4),
                        show=False,
                    )
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

            keypoints_3d = []
            for i in range(points.shape[0]):
                if np.isfinite(points[i, 0]) and np.isfinite(points[i, 1]):
                    x, y = points[i, 0].astype(int), points[i, 1].astype(int)
                    z = self.camera.get_depth_from_pixels((min(x, depth_img.shape[1] - 1), min(y, depth_img.shape[0] - 1)), depth_img)
                    # z = depth_img[min(y, depth_img.shape[0] - 1), min(x, depth_img.shape[1] - 1)] * self.camera.depth_scale
                    z = z.item() * self.camera.depth_scale
                else:
                    x, y, z = np.nan, np.nan, np.nan
                keypoints_3d.append([x, y, z])
            key_points_mat[count, :-4, :] = self.camera.get_markers_pos_in_meter(np.array(keypoints_3d)).T
            key_points_mat[count, -4:, :] = np.stack([thorax_spheres[i].get_center() for i in range(len(thorax_spheres))], axis=0)
            cup_points_mat[count, :] = center_meters.flatten()
            count += 1
        self.keypoints_3d = key_points_mat.copy()
        self.cup_points = cup_points_mat.copy()
        return key_points_mat, cup_points_mat

    def _prepare_thorax_icp(self):
        shoulder_pos = self.keypoints[
            0, [self.wholebody.get_index("right_shoulder"), self.wholebody.get_index("left_shoulder")], :
        ]
        mid_point, unit_vector = get_mid_point(shoulder_pos, 0.8)
        mid_point = mid_point.astype(int)
        d_mid_point = self.init_depth[mid_point[1], mid_point[0]]
        limit_depth = [
            d_mid_point - (0.08 * (1 / self.camera.depth_scale)),
            d_mid_point + (0.08 * (1 / self.camera.depth_scale)),
        ]
        distance = (
            np.linalg.norm([(shoulder_pos[1, 0] - shoulder_pos[0, 0]), (shoulder_pos[1, 1] - shoulder_pos[0, 1])]) * 0.8
        )
        virtual_markers = []
        distance_list = [distance * 0.3, distance * 0.3, distance * 0.6, distance * 0.6]
        ref_copy = shoulder_pos.copy()
        ref_copy[0, 0] += 35
        ref_copy[1, 0] -= 50
        refs = [ref_copy[0], ref_copy[1], ref_copy[0], ref_copy[1]]
        for i in range(len(distance_list)):
            pos_tmp = refs[i]
            virtual_markers.append((pos_tmp + distance_list[i] * unit_vector).tolist())

        shoulder_pos[0, 0] -= 30
        shoulder_pos[1, 0] += 30
        shoulder_pos[:, 1] -= 50
        crops_pos = shoulder_pos.copy().tolist()
        for i in [1, 0]:
            pos_tmp = shoulder_pos[i]
            crops_pos.append((pos_tmp + distance * 1.8 * unit_vector).tolist())

        spheres = []
        for marker in virtual_markers:
            x, y = np.array(marker).astype(int)
            cx, cy = self.camera.depth.ppx, self.camera.depth.ppy
            fx, fy = self.camera.depth.fx, self.camera.depth.fy
            # z = self.init_depth[y, x] 
            z = self.camera.get_depth_from_pixels((x, y), self.init_depth)

            Z = z * self.camera.depth_scale
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            mid_point_3d = np.array([X, Y, Z])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.050)
            sphere.translate(mid_point_3d)
            sphere.paint_uniform_color([1, 0, 0])
            spheres.append(sphere)
        self.thorax_crops, self.thorax_limit_depth, self.thorax_spheres = crops_pos, limit_depth, spheres

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
        cluster_base_filter=True,
        lp_filter=True,
        lp_filter_cutoff=4,
        lp_filter_order=2,
        cluster_eps_list=[0.1, 0.08, 0.06, 0.04, 0.02, 0.01],
        idxs_for_clustering=None,
    ):
        post_process_3d = self.keypoints_3d.copy()
        if remove_outliers_on_diff:
            post_process_3d = remove_outliers(self.keypoints_3d, on_diff=True)
        if cluster_base_filter:
            post_process_3d = clean_trajectory(post_process_3d, idxs=idxs_for_clustering, eps_list=cluster_eps_list)
        if lp_filter:
            post_process_3d = filter_points_3d(
                post_process_3d, lp_filter_cutoff, self.camera.color.fps, order=lp_filter_order
            )
        self.post_process_3d = post_process_3d
        return post_process_3d

    def save(self, export_trc=False):
        dic_to_save = {
            "keypoints_3d": self.post_process_3d,
            "keypoints_2d": self.keypoints,
            "idxs": self.idxs,
            "depth_image_path": self.depth_img_path,
            "color_image_path": self.color_img_path,
            "key_points_idxs": self.wholebody.minimal_idxs,
            "key_points_names": self.wholebody.minimal_set + [f"virtual_marker_{i}" for i in range(4)],
            "cup_points": self.cup_points,
        }
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


if __name__ == "__main__":
    file_dir = r"D:\Documents\Programmation\markerless_drinking_task\videos"
    dirs = os.listdir(file_dir)
    dirs = [os.path.join(file_dir, d) for d in dirs if os.path.isdir(os.path.join(file_dir, d))]
    processor = Keypoints3DProcessor()
    for dir in dirs:
        file_path = os.path.join(dir, "annotated", "keypoints.npy")
        depth_image_path = dir
        processor.initialize_data(file_path, dir, os.path.join(dir, "camera_config.json"))
        processor.compute_3d_coordinates(track_thorax=True, track_cup=False)
        processor.post_process(remove_outliers_on_diff=False)
        processor.save(export_trc=True)
