import numpy as np
import json
from scipy.spatial.transform import Rotation

try:
    import pyrealsense2 as rs

    rs_package = True
except ImportError:
    rs_package = False
    pass
    # print ImportWarning("Cannot use camera: Import of the library pyrealsense2 failed")


class CameraIntrinsics:
    def __init__(self):
        self.fps = None
        self.width = None
        self.height = None
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        self.intrinsics_mat = None
        self.model = None
        self.translation = None
        self.rotation = None
        self.dist_coefficients = None

    def set_intrinsics_from_file(self, fx_fy, ppx_ppy, dist_coefficients, size, fps):
        self.height = size[1]
        self.width = size[0]

        self.fx = fx_fy[0]
        self.fy = fx_fy[1]

        self.ppx = ppx_ppy[0]
        self.ppy = ppx_ppy[1]

        self.dist_coefficients = dist_coefficients
        self.model = rs.distortion.inverse_brown_conrady if rs_package else None

        self.fps = fps

        self._set_intrinsics_mat()

    def set_extrinsics_from_file(self, rotation, translation):
        self.rotation = rotation
        self.translation = translation

    def set_intrinsics(self, intrinsics):
        self.width = intrinsics.width
        self.height = intrinsics.height
        self.ppx = intrinsics.ppx
        self.ppy = intrinsics.ppy
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.dist_coefficients = intrinsics.coeffs
        self.model = intrinsics.model

        self._set_intrinsics_mat()

    def _set_intrinsics_mat(self):
        self.intrinsics_mat = np.array(
            [
                [self.fx, 0, self.ppx],
                [0, self.fy, self.ppx],
                [0, 0, 1],
            ],
            dtype=float,
        )

    def get_intrinsics(self, model=None):
        _intrinsics = rs.intrinsics()
        _intrinsics.width = self.width
        _intrinsics.height = self.height
        _intrinsics.ppx = self.ppx
        _intrinsics.ppy = self.ppy
        _intrinsics.fx = self.fx
        _intrinsics.fy = self.fy
        _intrinsics.coeffs = self.dist_coefficients
        _intrinsics.model = self.model
        if model:
            _intrinsics.model = model

        return _intrinsics
    
    def get_extrinsics(self):
        _extrinsics = rs.extrinsics()
        _extrinsics.rotation = self.rotation
        _extrinsics.translation = self.translation
        return _extrinsics



class CameraConverter:
    """
    CameraConverter class init the camera intrinsics to
    calculate the position of the markers in pixel
    via method 'express_in_pixel' or in meters
    via method 'get_markers_pos_in_meters'.
    """

    def __init__(self, use_camera: bool = False, model=None):
        """
        Init the Camera and its intrinsics. You can determine
        if you are using a camera to get the intrinsics via
        the 'use_camera parameter'. As well, you can change the
        default method of image distortion with 'model' parameter.

        Parameters
        ----------
        use_camera: bool
            True if you get the intrinsics via a connected camera.
            False if you get the intrinsics via configuration files.
        model: rs.intrinsics.property
            Model for distortion to apply on the image for the computation.
        """
        # Camera intrinsics
        self.depth = CameraIntrinsics()
        self.color = CameraIntrinsics()
        self.accel = CameraIntrinsics()
        self.model = model
        self.set_intrinsics = self._set_intrinsics_from_file if not use_camera else self._set_intrinsics_from_pipeline
        self.set_extrinsics = self._set_extrinsic_from_file if not use_camera else self._set_extrinsic_from_pipeline
        # Camera extrinsic
        self.depth_to_color = None
        self.conf_data_dic = None
        self.depth_scale = None
        self.camera_name = None
        self.accel_data = []
        self.accel_rotation = None

    def _set_intrinsics_from_file(self, conf_data: dict):
        """
        Private method.
        Set the Camera intrinsics from file and frame.

        Parameters
        ----------
        conf_data: dict
            Dictionary containing the values to init the intrinsics of the camera.
        """
        conf_data = load_json(conf_data)
        self.camera_name = conf_data["camera_name"]
        self.conf_data_dic = conf_data
        self.depth_scale = conf_data["depth_scale"]
        self.depth.set_intrinsics_from_file(
            conf_data["depth_fx_fy"],
            conf_data["depth_ppx_ppy"],
            conf_data["dist_coeffs_color"],
            conf_data["size_depth"],
            conf_data["depth_rate"],
        )
        self.color.set_intrinsics_from_file(
            conf_data["color_fx_fy"],
            conf_data["color_ppx_ppy"],
            conf_data["dist_coeffs_color"],
            conf_data["size_color"],
            conf_data["color_rate"],
        )
        if "accel_rate" in conf_data:
            self.accel.fps = conf_data["accel_rate"]

    def _set_extrinsic_from_file(self, conf_data=None):
        conf_data = self.conf_data_dic if not conf_data else load_json(conf_data)
        self.depth.set_extrinsics_from_file(conf_data["depth_to_color_rot"], conf_data["depth_to_color_trans"])
        if "accel_to_depth_rot" in conf_data and "accel_to_depth_trans" in conf_data:
            self.accel.set_extrinsics_from_file(conf_data["accel_to_depth_rot"], conf_data["accel_to_depth_trans"])
        if "accel_rotation" in conf_data and conf_data["accel_rotation"] is not None:
            self.accel_rotation = np.array(conf_data["accel_rotation"])
        # self.color.set_extrinsics_from_file(conf_data["color_to_depth_rot"], conf_data["color_to_depth_trans"])

    def _set_extrinsic_from_pipeline(self, pipeline):
        depth_profile = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
        accel_profile = pipeline.get_active_profile().get_stream(rs.stream.accel).as_motion_stream_profile()

        extrin = depth_profile.get_extrinsics_to(color_profile)
        self.depth.set_extrinsics_from_file(extrin.rotation, extrin.translation)

        extrin = color_profile.get_extrinsics_to(depth_profile)
        self.color.set_extrinsics_from_file(extrin.rotation, extrin.translation)

        if accel_profile is not None:
            extrin = accel_profile.get_extrinsics_to(depth_profile)
            self.accel.set_extrinsics_from_file(extrin.rotation, extrin.translation)


    def _set_intrinsics_from_pipeline(self, pipeline):
        """
        Private method.
        Set the Camera intrinsics from pipeline.

        Parameters
        ----------
        pipeline: Any
            Pipeline linked to the connected camera.
        """
        self.camera_name = pipeline.get_active_profile().get_device().get_info(rs.camera_info.name)
        _intrinsics = (
            pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics(), 
            pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        )
        self.depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        self.depth.set_intrinsics(_intrinsics[0]) 
        self.color.set_intrinsics(_intrinsics[1])
        self.depth.model = _intrinsics[0].model.name
        self.color.model = _intrinsics[1].model.name
        self.color.fps = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().fps()
        self.depth.fps = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().fps()
        self.accel.fps = pipeline.get_active_profile().get_stream(rs.stream.accel).as_motion_stream_profile().fps()

    def get_marker_pos_in_pixel(self, marker_pos_in_meters: np.array):
        """
        Get the intrinsics and compute the markers positions
        in meters to get the markers positions in pixel.

        Parameters
        ----------
        marker_pos_in_meters: np.array
            Markers positions in meters.

        Returns
        -------
        np.array
        """
        _intrinsics = self.depth.get_intrinsics(self.model)
        markers = []

        for i in range(len(marker_pos_in_meters)):
            computed_pos = rs.rs2_project_point_to_pixel(
                _intrinsics,
                np.array(
                    [marker_pos_in_meters[i][0], marker_pos_in_meters[i][1], marker_pos_in_meters[i][2]],
                    dtype=np.float32,
                ),
            )

            markers.append(computed_pos)
        markers[0][0] = np.array(markers[0][0]).clip(0, self.depth.width)
        markers[0][1] = np.array(markers[0][1]).clip(0, self.depth.height)
        markers[0][0] = 0 if np.isnan(markers[0][0]) else markers[0][0]
        markers[0][1] = 0 if np.isnan(markers[0][1]) else markers[0][1]

        return np.array(markers, dtype=np.int64)

    def get_markers_pos_in_meter(self, marker_pos_in_pixel: np.array):
        """
        Get the intrinsics and compute the markers positions
        in pixels to get the markers positions in meters.
        If both parameters are set then the given
        'marker_pos_in_pixel' override the one get from
        the method.

        Parameters
        ----------
        marker_pos_in_pixel: np.array
            Markers positions in meters.

        Returns
        -------
        np.array
        """
        _intrinsics = self.depth.get_intrinsics(self.model)
        # markers_in_meters = self._compute_markers(_intrinsics, marker_pos_in_pixel, rs.rs2_deproject_pixel_to_point)
        if max(marker_pos_in_pixel[2]) > 20:
            # print("The depth is not in meter. Applying the depth scale to convert it to meter.")
            marker_pos_in_pixel[:, 2] *= self.depth_scale 

        markers = [[], [], []]

        for i, pos in enumerate(marker_pos_in_pixel):
            if np.isnan(pos[0]) or np.isnan(pos[1]) or np.isnan(pos[2]):
                markers[0].append(np.nan)
                markers[1].append(np.nan)
                markers[2].append(np.nan)
                continue
            computed_pos = rs.rs2_deproject_pixel_to_point(
                _intrinsics, np.array([pos[0], pos[1]], dtype=np.float32), float(pos[2])
            )

            markers[0].append(computed_pos[0])
            markers[1].append(computed_pos[1])
            markers[2].append(computed_pos[2])

        return np.array(markers)

    def get_depth_from_pixels(self, pixels_pos, depth, neighbourhood=0, in_meter=True):
        # if depth is 0 average the depth in a neighbourhood of 5 pixels around the pixel position
        if not isinstance(pixels_pos, (list, tuple, np.ndarray)):
            pixels_pos = np.array(pixels_pos, dtype=np.int64)
        z = np.ndarray(pixels_pos.shape[0])
        for i in range(pixels_pos.shape[0]):
            if np.isnan(pixels_pos[i, 0]) or np.isnan(pixels_pos[i, 1]):
                z[i] = np.nan
                continue
            x = min(pixels_pos[i, 0], depth.shape[1] - 1)
            y = min(pixels_pos[i, 1], depth.shape[0] - 1)
            z[i] = depth[y, x].item()
            if z[i] == 0 and neighbourhood > 0:
                x = min(pixels_pos[i, 0], depth.shape[1] - neighbourhood - 1)
                y = min(pixels_pos[i, 1], depth.shape[0] - neighbourhood - 1)
                depth_non_zero = depth[y - neighbourhood : y + neighbourhood, x - neighbourhood : x + neighbourhood]
                z[i] = np.mean(depth_non_zero[depth_non_zero != 0])
            elif z[i] == 0:
                z[i] = np.nan
        if in_meter:
            z *= self.depth_scale
        return z

    def get_markers_pos_3d(
        self, marker_pos_in_pixel: np.array, depth: np.array, neighbourhood=5, in_pixel=True, depth_in_meter=True
    ):
        """
        Get the 3D markers positions. If 'in_pixel' is True, the markers positions are in pixel except for the depth, else they are in meters.
        Parameters
        ----------
        marker_pos_in_pixel: np.array
            Markers positions in meters.

        Returns
        -------
        np.array
        """
        if not in_pixel:
            markers_pixel = self.get_markers_pos_3d(
                marker_pos_in_pixel, depth, neighbourhood, in_pixel=True, depth_in_meter=False
            )
            return self.get_markers_pos_in_meter(markers_pixel)
        depth = self.get_depth_from_pixels(marker_pos_in_pixel, depth, neighbourhood, in_meter=depth_in_meter)
        return np.hstack([marker_pos_in_pixel, depth[:, None]])    

    @staticmethod
    def _compute_markers(
        intrinsics,
        marker_pos,
        method,
    ):
        """
        Private method.
        Compute the markers positions with the given method and intrinsics.
        For positions in meters to pixels use rs.rs2_project_point_to_pixel
        For positions in pixels to meters use rs.rs2_deproject_pixel_to_point

        Parameters
        ----------
        intrinsics: rs.intrinsics
            Camera intrinsics.
        marker_pos:
            Markers positions.
        method:
            Method to compute markers positions.

        Returns
        -------
        np.array
        """
        markers = [[], [], []]

        for i, pos in enumerate(marker_pos):
            computed_pos = method(intrinsics, np.array([pos[0], pos[1]], dtype=np.float32), float(pos[2]))

            markers[0].append(computed_pos[0])
            markers[1].append(computed_pos[1])
            markers[2].append(computed_pos[2])

        return np.array(markers)
    
    def save_config(self, path):
        if len(self.accel_data) > 0:
            self._compute_accel_matrix()

        self.conf_data_dic = {
            "camera_name": self.camera_name,
            "depth_scale": self.depth_scale,
            "depth_fx_fy": [self.depth.fx, self.depth.fy],
            "depth_ppx_ppy": [self.depth.ppx, self.depth.ppy],
            "color_fx_fy": [self.color.fx, self.color.fy],
            "color_ppx_ppy": [self.color.ppx, self.color.ppy],
            "depth_to_color_trans": self.depth.translation,
            "depth_to_color_rot": self.depth.rotation,
            "color_to_depth_trans": self.color.translation,
            "color_to_depth_rot": self.color.rotation,
            "model_color": self.color.model,
            "model_depth": self.depth.model,
            "dist_coeffs_color": self.color.dist_coefficients,
            "dist_coeffs_depth": self.depth.dist_coefficients,
            "size_color": [self.color.width, self.color.height],
            "size_depth": [self.depth.width, self.depth.height],
            "color_rate": self.color.fps,
            "depth_rate": self.depth.fps,
            "accel_rate": self.accel.fps,
            "accel_to_depth_trans": self.accel.translation,
            "accel_to_depth_rot": self.accel.rotation,
            "accel_rotation": self.accel_rotation.tolist() if self.accel_rotation is not None else None
        }
        with open(path, "w") as json_file:
            return json.dump(self.conf_data_dic, json_file, indent=4)
        
    def add_accel_frame(self, frame):
        accel = frame.first_or_default(rs.stream.accel)
        a = accel.as_motion_frame().get_motion_data()
        self.accel_data.append(np.array([a.x, a.y, a.z]))

    def _compute_accel_matrix(self):
        g = np.mean(np.array(self.accel_data), axis=0)
        g /= np.linalg.norm(g)
        target = np.array([0, 0, -1])      # Z upward convention

        axis = np.cross(g, target)
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-8:
            axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(g, target), -1, 1))
            R = Rotation.from_rotvec(axis * angle).as_matrix()
        else:
            R = np.eye(3)
        self.accel_rotation = R

    def align_with_z(self, points):
        if self.accel_rotation is None:
            raise ValueError("Accelerometer rotation matrix is not computed. Please add accelerometer frames first.")
        return np.dot(self.accel_rotation, points.T).T
        # return points @ self.accel_rotation

def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)
    

