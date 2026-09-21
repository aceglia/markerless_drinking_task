import os
from pathlib import Path

import numpy as np
from .processor import Processor

from ..rtmlib_utils import get_wholebody_model
from ..options import RGBDProcessingOptions

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

    def process_single_file(self, bag_file_path, options_file_path=None):
        if options_file_path is not None and options_file_path != "":
            self.apply_options(options_file_path)
        if self.wholebody_model is None:
            self.init_model()
        # base_dir, image_dir = process_bag_file(
        #     bag_file_path, self.wholebody_model, create_anotated_images=self.options.CNN_model["save_annotated_images"]
        # )
        base_dir = os.path.join(os.path.dirname(bag_file_path), Path(bag_file_path).stem) or '.'
        image_dir = os.path.join(base_dir, "images")
        self.keypoints_processor = Keypoints3DProcessor()
        self.keypoints_processor.initialize_data(
            base_dir + "/results/keypoints.npy", image_dir, os.path.join(base_dir, "camera_config.json"), show_pc=False
        )
        self.keypoints_processor.compute_3d_coordinates(track_thorax=True, track_cup=False)
        self.keypoints_processor.post_process(
            remove_outliers_on_diff=self.options.jump_filtering['remove_outliers_on_diff'],
            save_plot=self.options.jump_filtering['save_plot'],
            remove_outliers_on_sd=self.options.jump_filtering['remove_outliers_on_sd'],
            cluster_base_filter=self.options.jump_filtering['cluster_base_filter'],
        )
        self.keypoints_processor.save(export_trc=True)
        keypoints_3d = self.keypoints_processor.post_process_3d
        names = self.keypoints_processor.keypoints_names
        markers = np.array(keypoints_3d).T
        osim_model = r'opensim/wu_modified_markerless_left.osim'
        q = self._compute_kinematics(markers, names, osim_model, output_dir=base_dir + "/results")
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


