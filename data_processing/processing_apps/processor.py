import os

import numpy as np
from scipy.signal import butter, filtfilt
from data_processing.UKF import JointMarkerUKF
from data_processing.motion_segmentation import MotionSegmentation
from multiprocessing import Pool


class Processor:
    def __init__(self):
        self.opensim_model = None
        self.scale_model = False
        self.model_scaled = None
        self.options = None
        self.trial_files = []
        self.trials_data = []
        self.processed_trials = []

    def scale_model(self):
        pass

    def process_trials(self, path_list, **kwargs):
        for path in path_list:
            self.process_single_file(path, **kwargs)
        # cpu_count = os.cpu_count()
        # num_processes = max(1, cpu_count - 2)

        # args = [(path, kwargs) for path in path_list]

        # with Pool(processes=num_processes) as pool:
        #     results = pool.map(self.process_single_file, args, chunksize=1)


    def _apply_event_cut(self, markers_data, marker_names):
        if not self.options.cut_from_events or self.options.paired_event_idx is None:
            return markers_data, marker_names
        paired_event_idx = self.options.paired_event_idx
        markers_data_list = []
        for start_idx, stop_idx in paired_event_idx:
            markers_data_list.append(markers_data[:, :, start_idx:stop_idx])
        markers_data = np.stack(markers_data_list)
        return markers_data, marker_names

    def _apply_fill_gaps(self, markers_data, marker_names):
        if not self.options.fill_gaps:
            return markers_data, marker_names
        for i, name in enumerate(marker_names):
            mask = np.isnan(markers_data[0, i, :])
            padded_mask = np.concatenate(([False], mask, [False]))
            idx_diff = np.diff(padded_mask.astype(int))
            starts = np.flatnonzero(idx_diff == 1)
            ends = np.flatnonzero(idx_diff == -1)
            idx = np.arange(markers_data.shape[-1])
            for s, e in zip(starts, ends):
                if e - s >= self.options.fill_gaps_threshold:
                    continue
                for j in range(markers_data.shape[0]):
                    markers_data[j, i, s:e] = np.interp(idx[s:e], idx[~mask], markers_data[j, i, ~mask])
        return markers_data, marker_names

    def _apply_low_pass_filter(self, markers_data, marker_names):
        if not self.options.low_pass_filter:
            return markers_data, marker_names
        for i, name in enumerate(marker_names):
            mask = np.isfinite(markers_data[0, i, :])
            b, a = butter(self.options.filter_order, self.options.filter_cutoff / (0.5 * self.options.fs), btype="low")
            padded_mask = np.concatenate(([False], mask, [False]))
            idx_diff = np.diff(padded_mask.astype(int))
            starts = np.flatnonzero(idx_diff == 1)
            ends = np.flatnonzero(idx_diff == -1)
            min_lenght = 3 * (self.options.filter_order + 1)
            for s, e in zip(starts, ends):
                if e - s <= min_lenght:
                    continue
                for j in range(markers_data.shape[0]):
                    markers_data[j, i, s:e] = filtfilt(b, a, markers_data[j, i, s:e])
        return markers_data, marker_names

    def save_processed_trials(self):
        for i, (processed_data, processed_marker_names) in enumerate(self.processed_trials):
            trial_file = self.trial_files[i]
            base_name = os.path.splitext(os.path.basename(trial_file))[0]
            output_file = os.path.join(os.path.dirname(trial_file), f"{base_name}_processed.trc")
            self._save_trc(processed_data, processed_marker_names, output_file)

    def _save_trc(self, markers_data, marker_names, output_file):
        if not self.options.cut_from_events or self.options.paired_event_idx is None:
            markers_data = markers_data[None]
        for d, data in enumerate(markers_data):
            output_file = (
                output_file if markers_data.shape[0] == 1 else output_file.replace(".trc", f"_trial_{d+1}.trc")
            )
            self._save_single_trc(data, marker_names, output_file)

    def _save_single_trc(self, markers_data, marker_names, output_file):
        n_frames = markers_data.shape[-1]
        with open(output_file, "w") as f:
            f.write("PathFileType\t4\t(X/Y/Z)\t" + os.path.basename(output_file) + "\n")
            f.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
            )
            f.write(
                f"{60}\t{60}\t{n_frames}\t{len(marker_names)}\tm\t{60}\t1\t{n_frames}\n"
            )
            f.write("Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names]) + "\n")
            f.write(
                "\t\t"
                + "\t".join([f"{coord}{n + 1}\t" for n, name in enumerate(marker_names) for coord in ["X", "Y", "Z"]])
                + "\n"
            )
            for frame_idx in range(n_frames):
                time = frame_idx / 60
                f.write(f"{frame_idx + 1}\t{time:.5f}\t")
                for marker_idx in range(len(marker_names)):
                    x, y, z = markers_data[:, marker_idx, frame_idx]
                    f.write(f"{x:.5f}\t{y:.5f}\t{z:.5f}\t")
                f.write("\n")
        self._handle_nan(output_file)

    def _handle_nan(self, path):
        with open(path, "r") as file:
            data = file.read()
        data = data.replace("nan", "NaN")
        with open(path, "w") as file:
            file.write(data)
        return

    def _compute_kinematics(self, markers_data, marker_names, model_path, output_dir=None):
        markers = np.array([markers_data[:, marker_names.index(name), :] for name in marker_names]).transpose(1, 0, 2)
        self.ukf = JointMarkerUKF(
            model_path,
            data_rate=60,
            with_markers=False,
            type="constant_acceleration",
            experimental_marker_names=marker_names,
        )
        states = self.ukf.run(markers)
        self.ukf.save(os.path.join(output_dir, "kinematics.pkl"), {}, mot_file=True)
        return states

    def _compute_segmentation(
        self,
        expe_markers,
        experimental_marker_names,
        q,
        dof_names,
        side,
        camera=None,
        img_paths=None,
        threshold_onset=0.1,
        threshold_drinking=0.15,
        threshold_transporting=0.1,
        output_dir="",
    ):
        segmentation = MotionSegmentation(expe_markers, experimental_marker_names, q, dof_names, side=side)
        segmentation.perform_segmentation(
            threshold_onset=threshold_onset,
            threshold_drinking=threshold_drinking,
            threshold_transporting=threshold_transporting,
            img_paths=img_paths,
            camera=camera,
        )
        segmentation.plot(save_path=os.path.join(output_dir, "segmentation.png"))
        segmentation.save(os.path.join(output_dir, "segmentation.pkl"))
