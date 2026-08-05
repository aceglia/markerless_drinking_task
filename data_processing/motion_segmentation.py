import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from scipy.signal import find_peaks

from camera_converter import CameraConverter

try:
    from ultralytics import YOLO
except ImportError:
    pass


class MotionSegmentation:
    def __init__(self, expe_markers, experimental_marker_names, joint_angles=None, joint_names=None, fps=30, side='left'):
        self.expe_markers = expe_markers
        self.experimental_marker_names = experimental_marker_names
        self.joint_angles = joint_angles
        self.side = side
        self.joint_names = joint_names
        self.fps = fps
        self.time = np.arange(expe_markers.shape[-1]) / fps
        self._compute_velocity()
        self._yolo_model = None

    def _compute_velocity(self):
        self.markers_velocity = np.gradient(self.expe_markers, 1 / self.fps, axis=-1, edge_order=1)
        self.markers_acceleration = np.gradient(self.markers_velocity, 1 / self.fps, axis=-1, edge_order=1)
        self.markers_jerk = np.gradient(self.markers_acceleration, 1 / self.fps, axis=-1, edge_order=1)
        self.markers_jerk_norm = np.linalg.norm(self.markers_jerk, axis=0) * 1e3
        self.markers_acc_norm = np.linalg.norm(self.markers_acceleration, axis=0) * 1e3
        self.markers_speed_norm = np.linalg.norm(self.markers_velocity, axis=0) * 1e3
        self.wrist_marker = self.expe_markers[:, self.experimental_marker_names.index(f"{self.side}_wrist"), :]
        self.wrist_marker_speed_norm = self.markers_speed_norm[self.experimental_marker_names.index(f"{self.side}_wrist"), :]
        wrist_norm = np.linalg.norm(self.wrist_marker, axis=0)
        self.wrist_velocity = np.gradient(wrist_norm, 1 / 30, axis=-1, edge_order=1)
        self.nose_wrist_dist = np.linalg.norm(
            self.expe_markers[:, self.experimental_marker_names.index("nose"), :] - self.wrist_marker, axis=0
        )
        self.join_velocity = np.gradient(self.joint_angles, 1 / self.fps, axis=-1, edge_order=1)

    def get_onset_offset(self, threshold=0.05):
        wrist_tmp = abs(self.wrist_velocity)
        max_value = np.max(wrist_tmp, axis=0)
        idxs = np.where(wrist_tmp > (threshold * max_value))[0]
        if len(idxs) == 0:
            onset_idx = 0
        else:
            onset_idx = self._find_local_minima(wrist_tmp, idxs[0], window=30, backward=True)
        offset_idx = np.where(wrist_tmp > (threshold * max_value))[0]
        if len(idxs) == 0:
            offset_idx = len(self.wrist_marker_speed_norm) - 1
        else:
            offset_idx = self._find_local_minima(wrist_tmp, idxs[-1], window=30, backward=False)
            
        self.offset_idx = offset_idx
        self.onset_idx = onset_idx
        return onset_idx, offset_idx

    def _find_local_minima(self, data, detection, window=30, backward=True):
        # check if ascending or descending gradient to find local minima
        is_ascending = bool(np.all(np.diff(data[detection-2:detection+2]) >= 0))

        if backward:
            data_tmp = data[int(max(0, detection - window)):detection]
        else:
            data_tmp = data[detection:min(len(data), detection + window)]
        grad = -np.gradient(data_tmp) if not is_ascending else np.gradient(data_tmp)
        local_minima = np.where(grad < 0)[0]
        if backward:
            if len(local_minima) == 0:
                return int(max(0, detection - window))
            else:
                return local_minima[-1] + int(max(0, detection - window))
        else:
            if len(local_minima) == 0:
                return min(len(data), detection + window)
            else:
                return local_minima[0] + detection

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
        cup_center = None
        # check if multiple cups are detected and select the bigest bbox
        if len(cup_boxes) > 1:
            cup_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in cup_boxes]
            cup_centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in cup_boxes]
            largest_cup_idx = np.argmax(cup_areas)
            cup_center = np.array(cup_centers[largest_cup_idx]).astype(int)
        if len(cup_boxes) == 1:
            box = cup_boxes[0]
            cup_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]).astype(int)
        if cup_center is not None:
            cup_centers = cup_center 
        return cup_center

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
            if threshold > 0:
                idx_start = self._find_local_minima(center_to_hand_dist_start, idx_start, window=30, backward=True)
        if len(idx_end) == 0:
            idx_end = len(center_to_hand_dist_end) - 1
        else:
            idx_end = idx_end[0] + self.drinking_end_idx
            if threshold > 0:
                idx_end = self._find_local_minima(center_to_hand_dist_end, idx_end, window=30, backward=False)
        self.transport_start_idx = idx_start
        self.transport_end_idx = idx_end
        return (idx_start, idx_end)

    def get_cup_centers(self, img_paths, camera, cup_offset=0.05):
        cup_centers = []
        iterations = 30
        count = 0
        for event in [self.onset_idx, self.offset_idx]:
            while count < iterations:
                count += 1
                try:
                    color_img = cv2.imread(img_paths[0][event])
                    depth_img = cv2.imread(img_paths[1][event], cv2.IMREAD_ANYDEPTH)
                except Exception as e:
                    event += 1
                    continue
                cup_center = self._detect_cup(color_img)
                if cup_center is None:
                    event += 1
                    continue
                center_meters = camera.get_markers_pos_3d(cup_center[None], depth_img, 5, depth_in_meter=True, in_pixel=False)
                center_meters[-1] += cup_offset
                break
            cup_centers.append(center_meters)
            count = 0
        if len(cup_centers) > 1:
            self.cup_centers = camera.align_with_z(np.array(cup_centers).reshape(2, 3))
        else:
            self.cup_centers = cup_centers
        return self.cup_centers

    def _reorder_paths(self, img_paths):
        pattern = r"color_(\d+).png"
        img_paths[0].sort(key=lambda x: int(re.search(pattern, x).group(1)))
        pattern = r"depth_(\d+).png"
        img_paths[1].sort(key=lambda x: int(re.search(pattern, x).group(1)))
        return img_paths

    def _get_phase_time(self, idx_stat, idx_end):
        time_start = self._time_from_idx(idx_stat)
        time_end = self._time_from_idx(idx_end)
        return (time_end - time_start)

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
        self.segmentation_results = {
            "onset_idx": self.onset_idx,
            "offset_idx": self.offset_idx,
            "drinking_start_idx": self.drinking_start_idx,
            "drinking_end_idx": self.drinking_end_idx,
            "cup_centers": self.cup_centers if use_cup_tracking else None,
            "transport_start_idx": self.transport_start_idx,
            "transport_end_idx": self.transport_end_idx,
            "distance_cup": self.cup_center_start_to_end
        }
        return self.segmentation_results
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.segmentation_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def plot(self):
        fps = self.fps
        plt.rcParams['svg.fonttype'] = 'none'
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        time = np.arange(self.joint_angles.shape[1]) / fps
        axs[0].set_title("Distance between nose and wrist (mm)")
        axs[0].plot(time, self.nose_wrist_dist * 1e3, label="distance nose-wrist")
        axs[1].set_title("Wrist speed (mm/s)")
        # axs[1].plot(
        #     time, self.markers_speed_norm[experimental_marker_names.index(f"{self.side}_wrist")]
        # )

        axs[1].plot(time, self.wrist_velocity)

        axs[2].set_title("Cup to hand distance (mm)")
        axs[2].plot(
            time,
            np.linalg.norm(self.wrist_marker - self.cup_centers[0].reshape(3, 1), axis=0) * 1e3,
            label="cup to hand start",
        )
        axs[2].plot(
            time, np.linalg.norm(self.wrist_marker - self.cup_centers[1].reshape(3, 1), axis=0) * 1e3, label="cup to hand end"
        )
        axs[2].legend()
        # wrist_acc_norm = np.linalg.norm(wrist_acc, axis=0)
        # motion_energy = np.cumsum(np.sum(self.markers_speed_norm**2, axis=0))
        axs[3].set_title("Elbow_flexion (deg)")
        # axs[3].plot(time, np.linalg.norm(self.expe_markers[:, self.experimental_marker_names.index("left_shoulder")], axis=0), label="left shoulder")
        # axs[3].plot(time, motion_energy, label="left shoulder x")
        elb_flex_name = f'elbow_left_flexion' if self.side == 'left' else f'elbow_flexion'
        axs[3].plot(time, self.joint_angles[self.joint_names.index(elb_flex_name)] * 180 / np.pi, label=elb_flex_name)

        [ax.vlines(time[self.onset_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="red") for ax in axs]       
        [ax.vlines(time[self.offset_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="red") for ax in axs]
        [ax.vlines(time[self.drinking_start_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="green") for ax in axs]
        [ax.vlines(time[self.drinking_end_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="green") for ax in axs]
        [ax.vlines(time[self.transport_start_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="blue") for ax in axs]
        [ax.vlines(time[self.transport_end_idx], ax.get_ylim()[0], ax.get_ylim()[1], color="blue") for ax in axs]

        # put labels: 
        reaching_time = self._get_phase_time(self.onset_idx, self.transport_start_idx)
        transporting_one = self._get_phase_time(self.transport_start_idx, self.drinking_start_idx)
        drinking_time = self._get_phase_time(self.drinking_start_idx, self.drinking_end_idx)
        transport_time = self._get_phase_time(self.drinking_end_idx, self.transport_end_idx)
        motion_end = self._get_phase_time(self.transport_end_idx, self.offset_idx)

        [ax.text(time[self.onset_idx], ax.get_ylim()[1], f"Reaching: \n{reaching_time:.2f}s", color="red", verticalalignment="top") for ax in axs]
        [ax.text(time[self.offset_idx], ax.get_ylim()[1], f"Offset: {motion_end:.2f}s", color="red", verticalalignment="top") for ax in axs]
        [ax.text(time[self.drinking_start_idx], ax.get_ylim()[1], f"Drinking: \n{drinking_time:.2f}s", color="green", verticalalignment="top") for ax in axs]
        [ax.text(time[self.drinking_end_idx], ax.get_ylim()[1], f"Trans.: \n{transport_time:.2f}s", color="green", verticalalignment="top") for ax in axs]
        [ax.text(time[self.transport_start_idx], ax.get_ylim()[1], f"Trans.: \n{transporting_one:.2f}s", color="blue", verticalalignment="top") for ax in axs]
        [ax.text(time[self.transport_end_idx], ax.get_ylim()[1], f"Return: \n{motion_end:.2f}s", color="blue", verticalalignment="top") for ax in axs]

        plt.show(block=True)


if __name__ == "__main__":
    pickle_file = (
        r"D:\Documents\Programmation\markerless_drinking_task\videos\20260514_112042\annotated\ukf_results.pkl"
    )
    camera_config_file = (
        r"D:\Documents\Programmation\markerless_drinking_task\videos\20260514_112042\camera_config.json"
    )
    res = pickle.load(open(pickle_file, "rb"))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(res["keypoints_3d"][..., 0], res["keypoints_3d"][..., 1], res["keypoints_3d"][..., 2], c='r')
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
    segmentation = MotionSegmentation(expe_markers, experimental_marker_names, q, dof_names, side='left')
    segmentation.perform_segmentation(
        threshold_onset=0.08,
        threshold_drinking=0.15,
        threshold_transporting=0.05,
        img_paths=(res["color_image_path"], res["depth_image_path"]),
        camera=camera,
    )
    segmentation.plot()
