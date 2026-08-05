import numpy as np
import os

from data_processing.keypoints_processor import Keypoints3DProcessor
from data_processing.UKF import JointMarkerUKF
from data_processing.motion_segmentation import MotionSegmentation


if __name__ == "__main__":
    file_dir = r"D:\Documents\Programmation\markerless_drinking_task\videos"
    file_dir = r'F:\CIME_MS\adnane'
    
    file_dir = r"C:\Users\neuromobility_lab\Documents\CIME_MS\test_002"
    model_path = r"C:\Users\neuromobility_lab\Documents\amedeo\dev\markerless_drinking_task\opensim\wu_modified_markerless.osim"
    dirs = os.listdir(file_dir)
    dirs = [os.path.join(file_dir, d) for d in dirs if os.path.isdir(os.path.join(file_dir, d))]
    camera_frames_range = None
    processor = Keypoints3DProcessor(camera_frames_range=camera_frames_range)
    for dir in dirs:
        file_path = os.path.join(dir, "annotated", "keypoints.npy")
        file = file_path.replace("keypoints.npy", "keypoints_3d.pkl")
        # with open(file, "rb") as f:
        #     data = pickle.load(f)
        # depth_image_path = dir
        processor.initialize_data(file_path, dir, os.path.join(dir, "camera_config.json"), show_pc=False)
        processor.compute_3d_coordinates(track_thorax=True, track_cup=False)
        processor.post_process(remove_outliers_on_diff=False, plot=False, remove_outliers_on_sd=True, cluster_base_filter=True)
        processor.save(export_trc=True)
        keypoints_3d = processor.post_process_3d
        names = processor.keypoints_names
        markers = np.array(keypoints_3d).T
        ukf = JointMarkerUKF(
            model_path, data_rate=processor.camera.color.fps, with_markers=False, type="constant_acceleration", experimental_marker_names=names
        )
        states = ukf.run(markers)
        ukf.save(os.path.join(dir, "annotated", "ukf_results.pkl"), processor.to_dict(), mot_file=True)

        n_dofs = ukf.N_JOINTS
        q = ukf.states[:n_dofs]
        dq = ukf.states[n_dofs : 2 * n_dofs]
        dof_names = ukf.dof_names
        expe_markers = ukf.expe_markers
        estimated_markers = ukf.model_markers
        model_marker_names = ukf.model_marker_names
        experimental_marker_names = ukf.experimental_marker_names
        segmentation = MotionSegmentation(expe_markers, experimental_marker_names, q, dof_names, side=processor.side)
        segmentation.perform_segmentation(
            threshold_onset=0.1,
            threshold_drinking=0.15,
            threshold_transporting=0.1,
            img_paths=(processor.color_img_path, processor.depth_img_path),
            camera=processor.camera,
        )
        segmentation.save(os.path.join(dir, "annotated", "segmentation.pkl"))
        segmentation.plot()

