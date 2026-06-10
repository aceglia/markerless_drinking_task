import matplotlib
from sklearn.cluster import DBSCAN

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from camera_converter import CameraConverter
from model_type import WholeBody
import numpy as np
import os
import cv2
from scipy import signal
import pickle
from C3DtoTRC import WriteTrcFromMarkersData
import open3d as o3d
from ultralytics import YOLO
from projection_utils import get_thorax_pc, get_mid_point, get_vec_from_should, perform_icp



class Keypoints3DProcessor:
    def __init__(self, yolo_model_path="yolov11.pt"):
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None
        self.wholebody = WholeBody()

    def _get_yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO(self.yolo_model_path)
        return self._yolo_model

    def initialize_data(self, data_path, img_dir, camera):
        if isinstance(camera, str):
            camera_config_path = camera
            camera = CameraConverter()
            camera.set_intrinsics(os.path.join(dir, camera_config_path))
            camera.set_extrinsics(os.path.join(dir, data_path))
        keypoints, idxs = self.return_unique_keypoints(np.load(file_path))
        self.color_img_path = [os.path.join(img_dir, f"depth_{int(i)}.png") for i in idxs]
        self.depth_img_path = [os.path.join(img_dir, f"color_{int(i)}.png") for i in idxs]
        self.camera = camera
        self.keypoints = self.wholebody.get_minimal_keypoints(keypoints, 0)
        self.keypoints_names = self.wholebody.minimal_set
        self._init_img()
        self._prepare_thorax_icp()
        self._prepare_cup_icp()

    def _init_img(self):
        self.init_depth = cv2.imread(self.depth_img_path[0], cv2.IMREAD_ANYDEPTH)
        self.init_color = cv2.imread(self.color_img_path[0], cv2.COLOR_BGR2RGB)

    def detect_cup(self, color_img, keypoints=None, keypoints_names=None):
        if keypoints is not None:
            if keypoints_names is None:
                raise ValueError(
                    "keypoints and keypoints_names must be provided if keypoints is not provided, and left_wrist must be in keypoints_names"
                )
            wrists_idx = [
                keypoints_names.index("left_wrist"),
                keypoints_names.index("right_wrist"),
            ]
            wrists_pos = keypoints[wrists_idx, :]
            color_croped = color_img.copy()

        model = self._get_yolo_model()
        results = model(color_img)
        cup_boxes = []
        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls)] == "cup":
                    cup_boxes.append(box.xyxy[0].cpu().numpy())
        return cup_boxes

    def get_3d_coordinates(
        self, keypoints, depth_img_path, camera, shoulder_idx, color_img_path, keypoints_names=None
    ):
        # keypoints[keypoints == 0] = np.nan
        shoulder_idx = [
            keypoints_names.index("right_shoulder"),
            keypoints_names.index("left_shoulder"),
        ]

        shoulder_pos = keypoints[0, shoulder_idx, :]
        key_points_mat = np.empty((keypoints.shape[0], keypoints.shape[1] + 4, 3))
        init_pc = None
        count = 0
        transformation = np.eye(4)
        for points, img, color in zip(keypoints, depth_img_path, color_img_path):
            depth_img = cv2.imread(img, cv2.IMREAD_ANYDEPTH)
            color_img = cv2.imread(color, cv2.COLOR_BGR2RGB)
            self.detect_cup(color_img, keypoints=points, keypoints_names=keypoints_names)
            if count == 0:
                crops_pos, limit_depth, mid_point = self.get_crop_init(
                    depth_img, shoulder_pos, camera, color_img
                )
            pc_tmp = self.get_thorax_pc(depth_img, camera, color_img, crops_pos, limit_depth)
            if count == 0:
                init_pc = o3d.geometry.PointCloud(pc_tmp)
                pc_prev = init_pc
                init_spheres = mid_point
            if count == len(depth_img_path) - 1:
                last_pc = pc_tmp
                last_spheres = mid_point
            if count > 0:
                mid_point, transformation = self.perform_icp(
                    pc_prev, pc_tmp, mid_point, threshold=0.01, initial_guess=transformation, show=False
                )
            pc_prev = o3d.geometry.PointCloud(pc_tmp)
            keypoints_3d = []
            for i in range(points.shape[0]):
                if np.isfinite(points[i, 0]) and np.isfinite(points[i, 1]):
                    x, y = points[i, 0].astype(int), points[i, 1].astype(int)
                    z = depth_img[min(y, depth_img.shape[0] - 1), min(x, depth_img.shape[1] - 1)] * camera.depth_scale
                else:
                    x, y, z = np.nan, np.nan, np.nan
                keypoints_3d.append([x, y, z])
            key_points_mat[count, :-4, :] = camera.get_markers_pos_in_meter(np.array(keypoints_3d)).T
            key_points_mat[count, -4:, :] = np.stack([mid_point[i].get_center() for i in range(len(mid_point))], axis=0)

            count += 1
        # o3d.visualization.draw_geometries([init_pc, last_pc, *init_speres, *last_spheres])
        return key_points_mat

    def filter_points_3d(self, data, cutoff=10, fs=30, order=2):
        b, a = signal.butter(order, cutoff, "low", fs=fs)
        filtered_mat = np.zeros_like(data)
        t = np.arange(data.shape[0])
        data_interp = np.copy(data)
        for i in range(data.shape[1]):
            for j in range(3):
                valid = ~np.isnan(data_interp[:, i, j])
                len_val = valid.nonzero()[0].shape[0]
                if len_val == data_interp.shape[0]:
                    continue
                data_interp[:, i, j] = np.interp(t, t[valid], data_interp[valid, i, j])
            filtered_sig = signal.filtfilt(b, a, data_interp[:, i].T)
            filtered_mat[:, i, :] = filtered_sig.T
        return filtered_mat

    def write_trc(self, data, names, output_file_path, data_rate):
        "data: 3xmxn"
        WriteTrcFromMarkersData(
            output_file_path=output_file_path,
            markers=data,
            marker_names=names,
            data_rate=data_rate,
            cam_rate=data_rate,
            n_frames=data.shape[2],
            start_frame=1,
            units="m",
        ).write()

    def return_unique_keypoints(self, keypoints):
        idx = keypoints[:, 0, 3]
        _, i = np.unique(idx, return_index=True)
        keypoints = keypoints[i, :, :3]
        return keypoints, idx[i].astype(int)

    def _prepare_thorax_icp(self):
        shoulder_pos = self.keypoints[:, [self.wholebody.get_index("right_shoulder"), self.wholebody.get_index("left_shoulder")], :]
        mid_point = self.get_mid_point(shoulder_pos).astype(int)
        d_mid_point = self.init_depth[mid_point[1], mid_point[0]]
        limit_depth = [
            d_mid_point - (0.08 * (1 / self.camera.depth_scale)),
            d_mid_point + (0.08 * (1 / self.camera.depth_scale)),
        ]
        distance = np.linalg.norm(
            [(shoulder_pos[1, 0] - shoulder_pos[0, 0]), (shoulder_pos[1, 1] - shoulder_pos[0, 1])]
        ) * 0.8
        unit_vector = self.get_vec_from_should(shoulder_pos)

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
            z = self.init_depth[y, x] * self.camera.depth_scale
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            Z = z
            mid_point_3d = np.array([X, Y, Z])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.050)
            sphere.translate(mid_point_3d)
            sphere.paint_uniform_color([1, 0, 0])
            spheres.append(sphere)
        self.thorax_crops, self.thorax_limit_depth, self.thorax_spheres = crops_pos, limit_depth, spheres


if __name__ == "__main__":
    file_dir = r"videos"
    dirs = os.listdir(file_dir)
    dirs = [os.path.join(file_dir, d) for d in dirs if os.path.isdir(os.path.join(file_dir, d))]
    processor = Keypoints3DProcessor()
    for dir in dirs:
        # dir = r"D:\Documents\Programmation\markerless_drinking_task\videos\20260514_112323"
        file_path = os.path.join(dir, "annotated", "keypoints.npy")
        depth_image_path = dir
        # idxs = [int(im.replace("depth_", "").replace(".png", "")) for im in list_im]


        # cv2.imshow("col", color_img)
        # cv2.circle(color_img, np.array(mid_point, dtype=int), 2, [255, 0, 0], -1)
        # cv2.waitKey(0)
        # plt.scatter(*mid_point)
        keypoints_2d = wholebody.get_minimal_keypoints(keypoints, 0)
        should_idx = [wholebody.get_index("right_shoulder"), wholebody.get_index("left_shoulder")]
        keypoints_3d = processor.get_3d_coordinates(
            keypoints_2d, depth_path_list, camera, should_idx, color_path_list, wholebody.minimal_set
        )
        # keypoints_3d_filtered = processor.filter_points_3d(keypoints_3d, 5, camera.color.fps)
        # keypoints_3d_after = processor.remove_outliers(keypoints_3d, on_diff=True)
        keypoints_3d_after = keypoints_3d
        # keypoints_3d_after = processor.remove_outliers(keypoints_3d, on_diff=True)
        shoulder_idxs = [
            wholebody.get_index("right_shoulder"),
            wholebody.get_index("left_shoulder"),
            wholebody.get_index("right_elbow"),
            wholebody.get_index("left_elbow"),
        ]
        shoulder_idxs = list(range(keypoints_3d.shape[1] - 4))
        keypoints_3d_after = processor.clean_shoulder(keypoints_3d_after, shoulder_idxs)
        # keypoints_3d_filtered = processor.filter_points_3d(keypoints_3d, 5, camera.color.fps)

        keypoints_3d_filtered = processor.filter_points_3d(keypoints_3d_after, 4, camera.color.fps, order=2)

        dic_to_save = {
            "keypoints_3d": keypoints_3d_filtered,
            "keypoints_2d": keypoints_2d,
            "idxs": idxs,
            "depth_image_path": depth_image_path,
            "key_points_idxs": wholebody.minimal_idxs,
            "key_points_names": wholebody.minimal_set + [f"virtual_marker_{i}" for i in range(4)],
        }
        processor.write_trc(
            keypoints_3d_filtered.T,
            wholebody.minimal_set + [f"virtual_marker_{i}" for i in range(4)],
            file_path.replace("keypoints.npy", "keypoints_3d.trc"),
            camera.color.fps,
        )
        with open(file_path.replace("keypoints.npy", "keypoints_3d.pkl"), "wb") as f:
            pickle.dump(dic_to_save, f)
