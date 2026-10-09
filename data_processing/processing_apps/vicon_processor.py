import os
import pickle
import numpy as np
import ezc3d

from .processor import Processor
from ..options import ViconProcessorOptions


class Cluster:
    def __init__(self, name, cluster_markers, projected_markers):
        self.name = name
        self.cluster_markers = cluster_markers
        self.projected_markers = projected_markers
        self.local_markers = None
        self._local_marker_names = None
        self.local_markers_array = np.full((3, len(projected_markers)), np.nan)  # Initialize as empty array with shape (3, num_projected_markers, 0)
        self.missing_markers = []

    @staticmethod
    def get_local_cs(markers_data):
        M2, M1, M3 = markers_data[:3, 0, :], markers_data[:3, 1, :], markers_data[:3, 2, :]
        first_axis_vector = M3 - M1
        second_axis_vector = M2 - M1
        third_axis_vector = -np.cross(first_axis_vector, second_axis_vector, axis=0)
        second_axis_vector = np.cross(third_axis_vector, first_axis_vector, axis=0)
        n_frames = first_axis_vector.shape[1]
        rt = np.zeros((4, 4, n_frames))
        rt[:3, 0, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
        rt[:3, 1, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
        rt[:3, 2, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
        rt[:3, 3, :] = M1
        rt[3, 3, :] = 1
        return rt

    def compute_cs(self, markers_data, marker_names):
        marker_idx, names_found = self._get_safe_idx(marker_names, cluster=True)
        if names_found != self.cluster_markers:
            raise ValueError(f"Cluster '{self.name}' markers not found in the provided marker names. Found: {names_found}, Expected: {self.cluster_markers}")
        rt = self.get_local_cs(markers_data[:, marker_idx, :])
        return rt

    def _proj_local(self, markers_data, cs):
        if cs is None:
            raise ValueError("Local coordinate system not set for the cluster.")
        R, t = cs[:3, :3, :], cs[:3, 3:, :]
        transformed_markers = markers_data - t
        transformed_markers = np.moveaxis(transformed_markers, 2, 0)  # Move the first axis to the last position for matrix multiplication
        transformed_markers = R.T @ transformed_markers
        transformed_markers = np.moveaxis(transformed_markers, 0, 2)  # Move the last axis back to the first position
        return transformed_markers

    def _proj_global(self, markers_data, cs):
        cs = np.moveaxis(cs, 2, 0)  # Move the last axis to the first position for matrix multiplication
        R, t = cs[:, :3, :3], cs[:, :3, 3:]
        full_markers_data = np.tile(markers_data, (1, 1, cs.shape[0]))  # Repeat the markers data for each frame
        transformed_markers = np.moveaxis(full_markers_data, 2, 0) # Move the first axis to the last position for matrix multiplication
        transformed_markers = (R @ transformed_markers) + t
        transformed_markers = np.moveaxis(transformed_markers, 0, 2)  # Move the last axis back to the first position
        return transformed_markers

    def _get_safe_idx(self, marker_names, cluster=True):
        marker_list = self.cluster_markers if cluster else self.projected_markers
        marker_idx = []
        names_found = [name for name in marker_list if name in marker_names]
        for marker in names_found:
            marker_idx.append(marker_names.index(marker))
        return marker_idx, names_found

    def init_markers(self, markers_data, marker_names, overwrite=False):
        cs = self.compute_cs(markers_data, marker_names)
        marker_idx, names_found = self._get_safe_idx(marker_names, cluster=False)
        self._local_marker_names = names_found
        if len(marker_idx) < 1:
            raise ValueError(f"No projected markers found for cluster '{self.name}' in the provided marker names.")
        local_markers_tmp = np.full((3, len(self.projected_markers), markers_data.shape[2]), np.nan)  # Initialize as empty array with shape (3, num_projected_markers, n_frames)
        for name in self.projected_markers:
            if name in marker_names:
                idx = marker_names.index(name)
                local_markers_tmp[:, self.projected_markers.index(name), :] = markers_data[:, idx, :]
            else:
                self.missing_markers.append(name)
        self.missing_markers = np.unique(self.missing_markers).tolist()  # Remove duplicates
        if self.local_markers is None or overwrite:
            self.local_markers = self._proj_local(local_markers_tmp, cs)
        else:
            local_markers = self._proj_local(local_markers_tmp, cs)               
            self.local_markers = np.stack((self.local_markers.mean(axis=-1), local_markers.mean(axis=-1)), axis=-1)
        self.local_markers = np.nanmean(self.local_markers, axis=-1, keepdims=True)

    def _handle_missing_markers(self, markers_data, marker_names, cs):
        for name in self.missing_markers:
            if name in marker_names:
                idx = marker_names.index(name)
                marker_local = self._proj_local(markers_data[:, idx, :][:, np.newaxis, :], cs)
                self.local_markers[:, self.projected_markers.index(name), :] = np.nanmean(marker_local, axis=-1)
                self.missing_markers.remove(name)

    def get_markers(self, markers_data, marker_names):
        cs = self.compute_cs(markers_data, marker_names)
        if len(self.missing_markers) > 0:
            self._handle_missing_markers(markers_data, marker_names, cs)
        markers = self._proj_global(self.local_markers, cs)
        return markers


class ViconProcessor(Processor):
    def __init__(self, calibration_files=None):
        super().__init__()
        self.vicon_data = None
        self.calibration_files = calibration_files
        self.opensim_model = None
        self.scale_model = False
        self.model_scaled = None
        self.options = ViconProcessorOptions()
        self.clusters = {}
        self.is_initialized = False
        if self.calibration_files is not None:
            self.initialize()
        self.calibration_data = {}
        self.trial_files = []
        self.trials_data = []
        self.processed_trials = []

    def initialize(self, calibration_files=None, options_file=None, opensim_model=None, scale_tool=None):
        if calibration_files is not None and calibration_files != '':
            self.calibration_files = calibration_files
        if options_file is not None and options_file != '':
            self.options.from_file(options_file)

        self.opensim_model = opensim_model
        self.scale_tool = scale_tool

        for cluster_name, cluster_markers in self.options.clusters.items():
            if len(cluster_markers) != 2:
                print(f"WARNING: Cluster '{cluster_name}' must have exactly two lists: cluster markers and projected markers.")
                continue
            self.clusters[cluster_name] = Cluster(cluster_name, cluster_markers[0], cluster_markers[1])
            
        if self.calibration_files is not None and len(self.calibration_files) > 0:
            self._load_calibration_data()

    def _load_calibration_data(self):
        for file_path in self.calibration_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Calibration file not found: {file_path}")
            markers_data, marker_names = self._load_c3d(file_path)
            for cluster_name, cluster in self.clusters.items():
                if all(marker in marker_names for marker in cluster.cluster_markers):
                    cluster.init_markers(markers_data, marker_names, overwrite=False)
        self.is_initialized = True
        return True

    def batch_process_trials(self, trial_files, opensim_model=None, scale=None, rgbd_matching_files=None):
        self.trial_files = trial_files
        self.trials_data = []
        self.processed_trials = []
        for file_path in self.trial_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Trial file not found: {file_path}")
            markers_data, marker_names = self._load_c3d(file_path)
            rgbd_file = rgbd_matching_files.get(os.path.basename(file_path).replace('.c3d', ''), None) if rgbd_matching_files else None
            processed_data, processed_marker_names = self.process_trial(markers_data, marker_names, file_path, opensim_model=opensim_model, scale_model_file=scale, rgbd_file=rgbd_file)
            # self.trials_data.append((markers_data, marker_names))
            # self.processed_trials.append((processed_data, processed_marker_names))

    def _extract_markers_to_trc(self, markers_data, marker_names, file_path, out_dir=''):
        projected_data = {}
        for cluster_name, cluster in self.clusters.items():
            if all(marker in marker_names for marker in cluster.cluster_markers):
                projected_markers = cluster.get_markers(markers_data=markers_data, marker_names=marker_names)
                for i, proj_marker_name in enumerate(cluster.projected_markers):
                    projected_data[proj_marker_name] = projected_markers[:, i, :]
        processed_markers, processed_marker_names = self._apply_projection(projected_data, markers_data, marker_names)
        processed_markers, processed_marker_names = self._apply_fill_gaps(processed_markers, processed_marker_names)
        processed_markers, processed_marker_names = self._apply_low_pass_filter(processed_markers, processed_marker_names)
        processed_markers, processed_marker_names = self._apply_event_cut(processed_markers, processed_marker_names)
        saved_trc = self._save_trc(processed_markers, processed_marker_names, os.path.basename(file_path).replace(".c3d", ""), out_dir=out_dir, fps=self.options.fs) if self.options.export_trc else None
        return processed_markers, processed_marker_names, saved_trc

    def compute_kinematics(self, markers_data, marker_names, opensim_model, scale_model_file, output_dir, saved_trc=None, scale=True):
        if scale:
            marker_file = saved_trc
            output_model_path = output_dir + rf"/wu_modified_vicon_scaled.osim"
            try:
                self._scale_model(opensim_model, output_model_path, marker_file, setup_file=scale_model_file)
                scaled_model = output_model_path
            except Exception as e:
                print(f"Error occurred while scaling the model: {e}")
                scaled_model = opensim_model
        if not os.path.exists(output_model_path):
            scaled_model = opensim_model

        q = self._compute_kinematics(markers_data, marker_names, scaled_model, data_rate=100, output_dir=output_dir)
        side = 'left' if 'Left' in os.path.basename(output_dir) else 'right'
        self._compute_segmentation(
            markers_data,
            marker_names,
            q,
            self.ukf.dof_names,
            100,
            side,
            camera=None,
            img_paths=None,
            threshold_onset=self.options.motion_segmentation['threshold_onset'],
            threshold_drinking=self.options.motion_segmentation['threshold_drinking'],
            threshold_transporting=self.options.motion_segmentation['threshold_transporting'],
            output_dir=output_dir
        )
        return q
    
    def process_trial(self, markers_data, marker_names, file_path, opensim_model=None, scale_model_file=None, rgbd_file=None):
        out_dir = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace(".c3d", "_processed"))  # Create the output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        if not self.is_initialized:
            raise RuntimeError("ViconProcessor is not initialized. Please call 'initialize' with calibration files first.")
        processed_markers, processed_marker_names, saved_trc = self._extract_markers_to_trc(markers_data, marker_names, file_path, out_dir=out_dir)  # Process the trial and save the result
        count = 0
        for markers, trc in zip(processed_markers, saved_trc):
            scale = not 'scaled' in opensim_model and count == 0 
            q = self.compute_kinematics(markers, processed_marker_names, opensim_model, scale_model_file, output_dir=out_dir, saved_trc=trc, scale=scale)
            kin = self.load_rgbd_files(rgbd_file) if rgbd_file is not None else None
            kin = kin.T
            kin_resampled = np.interp(np.linspace(0, kin.shape[-1]-1, q.shape[-1]), np.arange(kin.shape[-1]), kin) if kin is not None else None
            if kin_resampled is not None:
                headers = self.ukf.dof_names
                to_csv = np.vstack((headers, kin_resampled))
                np.savetxt(os.path.join(out_dir, f"rgbd_100hz_kinematics_{count}.csv"), to_csv, delimiter=',')
            count += 1
        return processed_markers, processed_marker_names

    def load_rgbd_files(self, rgbd_file):
        path = os.path.join(rgbd_file, 'results/kinematics.csv')
        import pandas
        kinematics = pandas.read_csv(path)
        kinematics = kinematics.to_numpy()
        return kinematics

    def _apply_projection(self, projected_markers, markers_data, marker_names):
        for key, value in projected_markers.items():
            if key in marker_names:
                idx = marker_names.index(key)
                markers_data[:, idx, :] = value if self.options.projection['replace_existing'] else np.where(np.isnan(markers_data[:, idx, :]), value, markers_data[:, idx, :])
            else:
                marker_names.append(key)
                markers_data = np.concatenate((markers_data, value[:, np.newaxis, :]), axis=1)
        return markers_data, marker_names

    def _apply_event_cut(self, markers_data, marker_names):
        if not self.options.events['cut'] or self.options.paired_event_idx is None:
            return markers_data, marker_names
        paired_event_idx = self.options.paired_event_idx
        markers_data_list = []
        for start_idx, stop_idx in paired_event_idx:
            markers_data_list.append(markers_data[:, :, start_idx:stop_idx])
        markers_data = np.stack(markers_data_list)
        return markers_data, marker_names 

    def _load_c3d(self, file_path):
        data = ezc3d.c3d(file_path)
        markers_names_init = list(data["parameters"]["POINT"]["LABELS"]["value"])
        marker_names = [name for name in markers_names_init if "*" not in name]
        idx_markers = [markers_names_init.index(name) for name in marker_names]
        markers_data = data["data"]["points"][:3, idx_markers, :]
        unit = data["parameters"]["POINT"]["UNITS"]["value"][0]
        self.options.fs = data["header"]["points"]["frame_rate"]
        if unit != "m":
            markers_data /= 1000.0
        self.options.set_events(events=data["parameters"]["EVENT"], data_rate=self.options.fs)
        return markers_data, marker_names


if __name__ == "__main__":
    file_to_process = [r"C:\Users\neuromobility_lab\Documents\CIME_MS\test_002\vicon\Anato.c3d", 
                       r"C:\Users\neuromobility_lab\Documents\CIME_MS\test_002\vicon\thorax calibration.c3d"]
    trial_files = file_to_process + [r"C:\Users\neuromobility_lab\Documents\CIME_MS\test_002\vicon\DrinkingLeftArm.c3d", 
                   r"C:\Users\neuromobility_lab\Documents\CIME_MS\test_002\vicon\DrinkingRightArm.c3d"]
    options_file = r"C:\Users\neuromobility_lab\Documents\amedeo\dev\markerless_drinking_task\calibration_dict.yaml"
    processor = ViconProcessor()
    processor.initialize(calibration_files=file_to_process, options_file=options_file)
    processor.batch_process_trials(trial_files)
