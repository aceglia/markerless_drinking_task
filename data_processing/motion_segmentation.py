import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re

from camera_converter import CameraConverter

try:
    from ultralytics import YOLO
except ImportError:
    pass
from sklearn.cluster import DBSCAN


class MotionSegmentation:
    def __init__(self, expe_markers, experimental_marker_names, joint_angles=None, joint_names=None, fps=30):
        self.expe_markers = expe_markers
        self.experimental_marker_names = experimental_marker_names
        self.joint_angles = joint_angles
        self.joint_names = joint_names
        self.fps = fps
        self.time = np.arange(expe_markers.shape[-1]) / fps
        self._compute_velocity()
        self._yolo_model = None

    def _compute_velocity(self):
        markers_velocity = np.gradient(self.expe_markers, 1 / self.fps, axis=-1, edge_order=1)
        self.markers_speed_norm = np.linalg.norm(markers_velocity, axis=0) * 1e3
        self.wrist_marker = self.expe_markers[:, self.experimental_marker_names.index("left_wrist"), :]
        self.wrist_marker_speed_norm = self.markers_speed_norm[self.experimental_marker_names.index("left_wrist"), :]
        self.nose_wrist_dist = np.linalg.norm(
            self.expe_markers[:, self.experimental_marker_names.index("nose"), :] - self.wrist_marker, axis=0
        )
        self.join_velocity = np.gradient(self.joint_angles, 1 / self.fps, axis=-1, edge_order=1)

    def get_onset_offset(self, threshold=0.05):
        max = np.max(self.wrist_marker_speed_norm, axis=0)
        min = np.min(self.wrist_marker_speed_norm, axis=0)
        range = max - min
        onset_idx = np.where(self.wrist_marker_speed_norm - self.wrist_marker_speed_norm[0] > (threshold * range))[0]
        if len(onset_idx) == 0:
            onset_idx = 0
        else:
            onset_idx = onset_idx[0]
        offset_idx = np.where(self.wrist_marker_speed_norm - self.wrist_marker_speed_norm[-1] > (threshold * range))[0]
        if len(offset_idx) == 0:
            offset_idx = len(self.wrist_marker_speed_norm) - 1
        else:
            offset_idx = offset_idx[-1]
        self.offset_idx = offset_idx
        self.onset_idx = onset_idx
        return onset_idx, offset_idx

    def get_drinking_event(self, threshold=0.15):
        min_dist_idx = np.argmin(self.nose_wrist_dist)
        min_dist = self.nose_wrist_dist[min_dist_idx]
        range = np.max(self.nose_wrist_dist) - np.min(self.nose_wrist_dist)
        threshold = min_dist + threshold * range
        before_threshold_crossing = np.where(self.nose_wrist_dist[:min_dist_idx] > threshold)[0]
        if len(before_threshold_crossing) == 0:
            before_threshold_crossing = 0
        else:
            before_threshold_crossing = before_threshold_crossing[-1]

        after_threshold_crossing = np.where(self.nose_wrist_dist[min_dist_idx + 1 :] > threshold)[0]
        if len(after_threshold_crossing) == 0:
            after_threshold_crossing = len(self.nose_wrist_dist) - 1
        else:
            after_threshold_crossing = after_threshold_crossing[0] + min_dist_idx + 1
        self.drinking_start_idx = before_threshold_crossing
        self.drinking_end_idx = after_threshold_crossing
        return (before_threshold_crossing, after_threshold_crossing)

    def _time_from_idx(self, idx):
        return idx / self.fps

    def _get_yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO("yolo26s.pt")
        return self._yolo_model

    def _detect_cup(self, color_img):
        model = self._get_yolo_model()
        results = model(color_img)
        cup_boxes = []
        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls)] == "cup":
                    cup_boxes.append(box.xyxy[0].cpu().numpy())
        return cup_boxes

    def get_transporting_event(self, threshold=0.15):
        center_to_hand_dist_start = np.linalg.norm(self.wrist_marker - self.cup_centers[0].reshape(3, 1), axis=0)
        center_to_hand_dist_end = np.linalg.norm(self.wrist_marker - self.cup_centers[1].reshape(3, 1), axis=0)
        self.cup_center_start_to_end = np.linalg.norm(self.cup_centers[0] - self.cup_centers[1])
        min_start = np.min(center_to_hand_dist_start[self.onset_idx : self.drinking_start_idx])
        min_end = np.min(center_to_hand_dist_end[self.drinking_end_idx : self.offset_idx])
        idx_start = np.where(
            center_to_hand_dist_start[self.onset_idx : self.drinking_start_idx]
            <= min_start + threshold * min_start
        )[0]
        idx_end = np.where(
            center_to_hand_dist_end[self.drinking_end_idx : self.offset_idx]
            <= min_end + threshold * min_end
        )[0]
        if len(idx_start) == 0:
            idx_start = 0
        else:
            idx_start = idx_start[-1] + self.onset_idx
        if len(idx_end) == 0:
            idx_end = len(center_to_hand_dist_end) - 1
        else:
            idx_end = idx_end[0] + self.drinking_end_idx
        self.transport_start_idx = idx_start
        self.transport_end_idx = idx_end
        return (idx_start, idx_end)

    def get_cup_centers(self, img_paths, camera, cup_offset=0.05):
        cup_centers = []
        iterations = 30
        count = 0
        cup_boxes = []
        for event in [self.onset_idx, self.offset_idx]:
            while len(cup_boxes) == 0 and count < iterations:
                count += 1
                try:
                    color_img = cv2.imread(img_paths[0][event])
                    depth_img = cv2.imread(img_paths[1][event], cv2.IMREAD_ANYDEPTH)
                except Exception as e:
                    event += 1
                    continue
                cup_boxes = self._detect_cup(color_img)
                if len(cup_boxes) == 0:
                    event += 1
                    continue
                for box in cup_boxes:
                    center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    center = np.array(center).astype(int)
                    cup_depth = depth_img[center[1], center[0]] * camera.depth_scale
                    if cup_depth == 0:
                        depth_range = depth_img[center[1] - 10 : center[1] + 10, center[0] - 10 : center[0] + 10]
                        cup_depth = np.mean(depth_range[depth_range > 0]) * camera.depth_scale
                    center_meters = camera.get_markers_pos_in_meter([np.hstack([center, cup_depth - cup_offset])])
                    break
            cup_centers.append(center_meters)
            cup_boxes = []
            count = 0
        self.cup_centers = cup_centers
        return cup_centers

    def _reorder_paths(self, img_paths):
        pattern = r"color_(\d+).png"
        img_paths[0].sort(key=lambda x: int(re.search(pattern, x).group(1)))
        pattern = r"depth_(\d+).png"
        img_paths[1].sort(key=lambda x: int(re.search(pattern, x).group(1)))
        return img_paths

    def perform_segmentation(
        self,
        threshold_onset=0.05,
        threshold_drinking=0.15,
        threshold_transporting=0.08,
        use_cup_tracking=True,
        img_paths=None,
        camera=None,
    ):
        self.get_onset_offset(threshold=threshold_onset)
        self.get_drinking_event(threshold=threshold_drinking)
        if use_cup_tracking:
            img_paths = self._reorder_paths(img_paths)
            self.get_cup_centers(img_paths, camera, cup_offset=0.05)
            self.get_transporting_event(threshold=threshold_transporting)
        return {
            "onset_idx": self.onset_idx,
            "offset_idx": self.offset_idx,
            "drinking_start_idx": self.drinking_start_idx,
            "drinking_end_idx": self.drinking_end_idx,
            "cup_centers": self.cup_centers if use_cup_tracking else None,
            "transport_start_idx": self.transport_start_idx,
            "transport_end_idx": self.transport_end_idx,
            "distance_cup": self.cup_center_start_to_end
        }

    def plot(self, markers=[], dofs=[], velocity=[]):
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        time = np.arange(q.shape[1]) / 30
        axs[0].set_title("Distance between nose and wrist")
        axs[0].plot(time, self.nose_wrist_dist, label="distance nose-wrist")
        axs[1].set_title("Wrist speed")
        axs[1].plot(
            time, self.markers_speed_norm[experimental_marker_names.index("left_wrist")], label="left wrist exp"
        )
        axs[2].set_title("Cup to hand distance")
        axs[2].plot(
            time,
            np.linalg.norm(self.wrist_marker - self.cup_centers[0].reshape(3, 1), axis=0),
            label="cup to hand start",
        )
        axs[2].plot(
            time, np.linalg.norm(self.wrist_marker - self.cup_centers[1].reshape(3, 1), axis=0), label="cup to hand end"
        )
        axs[2].legend()
        wrist_acc = np.gradient(self.wrist_marker_speed_norm, 1 / self.fps, edge_order=1)
        # wrist_acc_norm = np.linalg.norm(wrist_acc, axis=0)
        # motion_energy = np.cumsum(np.sum(self.markers_speed_norm**2, axis=0))
        # axs[3].set_title("Shoulder marker")
        # axs[3].plot(time, np.linalg.norm(self.expe_markers[:, self.experimental_marker_names.index("left_shoulder")], axis=0), label="left shoulder")
        # axs[3].plot(time, motion_energy, label="left shoulder x")
        axs[3].plot(time, abs(wrist_acc), label="shoulder velocity")
        [ax.vlines(time[self.onset_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="red") for ax in axs]
        [ax.vlines(time[self.offset_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="red") for ax in axs]
        [ax.vlines(time[self.drinking_start_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="green") for ax in axs]
        [ax.vlines(time[self.drinking_end_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="green") for ax in axs]
        [ax.vlines(time[self.transport_start_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="blue") for ax in axs]
        [ax.vlines(time[self.transport_end_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="blue") for ax in axs]
        plt.show(block=True)


if __name__ == "__main__":
    pickle_file = (
        r"D:\Documents\Programmation\markerless_drinking_task\videos\20260514_112323\annotated\ukf_results.pkl"
    )
    camera_config_file = (
        r"D:\Documents\Programmation\markerless_drinking_task\videos\20260514_112323\camera_config.json"
    )
    res = pickle.load(open(pickle_file, "rb"))
    camera = CameraConverter()
    camera.set_intrinsics(camera_config_file)
    camera.set_extrinsics(camera_config_file)
    n_dofs = len(res["dof_names"])
    q = res["q"]
    dq = res["dq"]
    dof_names = res["dof_names"]
    expe_markers = res["experimental_markers"]
    estimated_markers = res["model_markers"]
    model_marker_names = res["model_marker_names"]
    experimental_marker_names = res["experimental_marker_names"]
    segmentation = MotionSegmentation(expe_markers, experimental_marker_names, q, dof_names)
    segmentation.perform_segmentation(
        threshold_onset=0.05,
        threshold_drinking=0.1,
        threshold_transporting=0,
        img_paths=(res["color_image_path"], res["depth_image_path"]),
        camera=camera,
    )
    segmentation.plot()