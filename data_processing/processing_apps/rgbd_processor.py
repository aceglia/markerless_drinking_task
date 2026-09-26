import os

import numpy as np
from .processor import Processor

from ..rtmlib_utils import get_wholebody_model
from ..options import RGBDProcessingOptions
from ..rtmlib_utils import process_bag_file

from ..keypoints_processor import Keypoints3DProcessor


class RGBDProcessor(Processor):
    def __init__(self, bag_file_path):
        super().__init__()
        self.bag_file_path = bag_file_path
        self.options = RGBDProcessingOptions()
        self.wholebody_model = None
        self.keypoints_processor = None
        self.fps = 60

    def init_model(self):
        self.wholebody_model = get_wholebody_model(
            device=self.options.CNN_model["device"], backend=self.options.CNN_model["backend"], mode=self.options.CNN_model["mode"], openpose_skeleton=False
        )

    def apply_options(self, options):
        self.options.from_file(options)
        self.init_model()

    def process_single_file(self, bag_file_path, options_file_path=None, run_pose_estimation=True, run_kinematics=True):
        if options_file_path is not None and options_file_path != "":
            self.apply_options(options_file_path)
        if self.wholebody_model is None:
            self.init_model()
        if run_pose_estimation:
            base_dir, image_dir = process_bag_file(
                bag_file_path, self.wholebody_model, create_anotated_images=self.options.CNN_model["save_annotated_images"]
            )
        else:
            base_dir = os.path.join(os.path.dirname(bag_file_path), os.path.splitext(os.path.basename(bag_file_path))[0])
            image_dir = os.path.join(base_dir, "images")
        if not os.path.exists(base_dir + "/results/keypoints.npy"):
            raise RuntimeError("Keypoints file not found. Please run pose estimation first.")
        self.keypoints_processor = Keypoints3DProcessor()
        self.keypoints_processor.initialize_data(
            base_dir + "/results/keypoints.npy", image_dir, os.path.join(base_dir, "camera_config.json"), show_pc=False
        )
        self.keypoints_processor.compute_3d_coordinates(track_thorax=True, track_cup=False)
        self.keypoints_processor.post_process(
            apply_smoother=self.options.jump_filtering['enable'],
            smoother_process_noise=self.options.jump_filtering['smooth_process_noise'],
            smoother_measurement_noise=self.options.jump_filtering['smooth_measurement_noise'],
            smoother_nis_threshold=self.options.jump_filtering['smooth_nis_threshold'],
            save_plot=self.options.jump_filtering['save_plot'],
            output_dir = base_dir + "/results"
        )
        self.keypoints_processor.save(export_trc=True)
        
        if run_kinematics:
            keypoints_3d = self.keypoints_processor.post_process_3d
            names = self.keypoints_processor.keypoints_names
            markers = np.array(keypoints_3d).T

            osim_model = rf'opensim/wu_modified_markerless_{self.keypoints_processor.side}.osim'
            setup_file = r'opensim/scaling_tool_markerless.xml'
            marker_file = base_dir + "/results/keypoints_3d.trc"
            output_model_path = base_dir + rf"/results/wu_modified_markerless_{self.keypoints_processor.side}_scaled.osim"
            try:
                self._scale_model(osim_model, output_model_path, marker_file, setup_file=setup_file)
                scaled_model = output_model_path
            except Exception as e:
                print(f"Error occurred while scaling the model: {e}")
                scaled_model = osim_model
            
            q = self._compute_kinematics(markers, names, scaled_model, output_dir=base_dir + "/results")
            self._compute_segmentation(
                markers,
                names,
                q,
                self.ukf.dof_names,
                self.keypoints_processor.side,
                camera=self.keypoints_processor.camera,
                img_paths=(self.keypoints_processor.color_img_path, self.keypoints_processor.depth_img_path),
                threshold_onset=self.options.motion_segmentation['threshold_onset'],
                threshold_drinking=self.options.motion_segmentation['threshold_drinking'],
                threshold_transporting=self.options.motion_segmentation['threshold_transporting'],
                output_dir=base_dir + "/results"
            )

    def _scale_model(self, osim_model_path, output_model_path, markers_file, setup_file):
        import opensim as osim
        osim_model = osim.Model(osim_model_path)
        scale_tool = osim.ScaleTool(setup_file)
        scale_tool.getGenericModelMaker().setModelFileName(osim_model_path)

        # Scale the model based on the marker data
        scaler = scale_tool.getModelScaler()
        scaler.setApply(True)
        scaler.setMarkerFileName(markers_file)
        scaler.setOutputModelFileName(output_model_path)
        scaler.processModel(osim_model) 

        # Place the markers on the scaled model
        scaled_model = osim.Model(output_model_path)
        marker_placer = scale_tool.getMarkerPlacer()
        marker_placer.setApply(True)
        marker_placer.setMarkerFileName(markers_file)
        marker_placer.setOutputModelFileName(output_model_path)
        marker_placer.processModel(scaled_model)
        return scaled_model