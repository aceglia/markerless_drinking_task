import sqlite3
import numpy as np
import os
import pyrealsense2 as rs
from .camera_converter import CameraConverter
from pathlib import Path
from functools import partial

#precalculate values needed to convert depth image to pointcloud
def prep_XY(depth_intrinsics: rs.intrinsics):

    #focal point and center of frame
    height = depth_intrinsics.height
    width = depth_intrinsics.width
    ppx = depth_intrinsics.ppx
    ppy = depth_intrinsics.ppy
    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy

    #create grid for point_cloud calculation
    points_xy = np.mgrid[0:height, 0:width].reshape((2, -1)).T

    #make the grid points centered
    points_xy = points_xy - np.array([ppy, ppx])
    points_x = points_xy[:,1]
    points_y = points_xy[:,0]

    #scaled_points
    points_x = points_x/fx
    points_y = points_y/fy

    return points_x, points_y


#this code converts depth image to pointcloud.
def depth_to_pointcloud(depth: np.ndarray, points_x: np.ndarray, points_y:np.ndarray, far_limit = -1) -> np.ndarray:

    '''
    far_limit is how far the points can be from the camera. 
    In this case, 2000 [milimeter] means points that are more than 2m are clamped to be 2m.
    If your application needs to see farther than 2 meter, then set this value higher.
    Or set to -1 if you dont want to use far_limit.
    '''

    Z = depth.flatten()
    if far_limit > 0:
        Z[Z > far_limit] = far_limit

    #calculate the x,y coordinates of the point cloud
    X = (points_x * Z)
    Y = (points_y * Z)

    #This is the pointcloud
    pcd = np.stack([X,Y,Z], axis = -1)

    return pcd

def align_manual(depth, points_x, points_y, depth_to_color_extrinsics, color_intrinsics):

    #convert the depth image to pointclound in depth camera coordinates
    pcd = depth_to_pointcloud(depth, points_x, points_y)

    #pointcloud in depth camera coordinates to pointcloud in color camera coordinates
    rot = np.array(depth_to_color_extrinsics.rotation).reshape((3,3))
    trans = np.array(depth_to_color_extrinsics.translation).reshape((1,3))
    #translation vector is in units of [meter] while pointcloud is in [milimeter]. So a convertion is necessary
    trans *= 1000.
    pcd_color = pcd @ rot + trans
    

    #Now the pointcloud exists in the coordinates defined where the color camera is at the origin.
    #Therefore, we simply project the pointcloud to image space using the camera intrinsics

    height = color_intrinsics.height
    width = color_intrinsics.width
    ppx = color_intrinsics.ppx
    ppy = color_intrinsics.ppy
    fx = color_intrinsics.fx
    fy = color_intrinsics.fy

    #split the pointcloud into its coordinates along each axis
    Xc = pcd_color[:,0]
    Yc = pcd_color[:,1]
    Zc = pcd_color[:,2]

    #these are indices of depth points that are valid. 
    nonzeros = Zc != 0

    #project to pixel coordinates. Only pointcloud points that are valid.
    #If you don't know how to project to camera pixels, look here:
    #https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#:~:text=Detailed%20Description
    #more specifically, the projection is done with this equation:
    #https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#:~:text=equivalent%20to%20the%20following
    u = fx * (Xc[nonzeros]/Zc[nonzeros]) + ppx
    v = fy * (Yc[nonzeros]/Zc[nonzeros]) + ppy
    #convert to image like indexing
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    #Now we are only interested in points that fall inside the color image frame
    u_inbound = (u >=0) * (u < width)
    v_inbound = (v >=0) * (v < height)
    inbound = u_inbound * v_inbound #these are indices of points that are inside the color camera frame
    
    #shortlist the pointcloud to only non-zero and in frame points
    u = u[inbound]
    v = v[inbound]
    z = Zc[nonzeros][inbound] #these are the depth values for each pixel

    #create a new frame for the aligned depth image
    manual_align = np.zeros((height, width), dtype = np.uint16)
    #apply the pixel values
    manual_align[(v, u)] = z

    return manual_align

def align_depth_to_color(
    depth_raw,
    depth_K,
    color_K,
    color_dist,
    R,
    t,
    depth_scale=0.001,
    color_shape=(480, 848)
):
    # -------------------------
    # Depth intrinsics
    # -------------------------
    depth_intr = rs.intrinsics()
    depth_intr.width = 848
    depth_intr.height = 480
    depth_intr.fx = depth_K[0, 0]
    depth_intr.fy = depth_K[1, 1]
    depth_intr.ppx = depth_K[0, 2]
    depth_intr.ppy = depth_K[1, 2]
    depth_intr.model = rs.distortion.none
    depth_intr.coeffs = [0, 0, 0, 0, 0]


    # -------------------------
    # Color intrinsics
    # -------------------------
    color_intr = rs.intrinsics()
    color_intr.width = 848
    color_intr.height = 480
    color_intr.fx = color_K[0, 0]
    color_intr.fy = color_K[1, 1]
    color_intr.ppx = color_K[0, 2]
    color_intr.ppy = color_K[1, 2]

    color_intr.model = rs.distortion.inverse_brown_conrady
    color_intr.coeffs = color_dist.tolist()

    # # -------------------------
    # # Depth -> Color extrinsics
    # # -------------------------
    # R = np.array([
    #     [ 0.999990, -0.000300,  0.003671],
    #     [ 0.000300,  1.000000,  0.000091],
    #     [-0.003671, -0.000090,  0.999990],
    # ], dtype=np.float32)

    # # Your current empirically better transform
    # t = np.array([
    #     -0.058996,
    #     -0.000019674,
    #     0.00049058
    # ], dtype=np.float32)

    depth_to_color = rs.extrinsics()
    depth_to_color.rotation = R.T.flatten().tolist()
    depth_to_color.translation = (t).tolist()
    manual_align = align_manual(depth_raw, *prep_XY(depth_intr), depth_to_color, color_intr)
    return manual_align


def init_db3(db3_path):
    pipeline = rs.pipeline()
    dir_path = os.path.dirname(db3_path) or '.'
    base_dir = os.path.join(dir_path, Path(db3_path).stem)
    os.makedirs(base_dir + '/results', exist_ok=True)

    tmp_path = os.path.join(base_dir, "images")
    os.makedirs(tmp_path, exist_ok=True)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 60)
    config.enable_stream(rs.stream.accel)
    rs.config.enable_device_from_file(config, db3_path, repeat_playback=False)
    profile = pipeline.start(config)
    converter = CameraConverter(use_camera=True)
    converter.set_intrinsics(pipeline)
    converter.set_extrinsics(pipeline)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    if converter.depth_scale == 0:
        print("WARNING: Something went wrong with the depth scale. Trying to load from db3 file...")
        depth_K, color_K, color_dist, R, t, depth_scale = load_realsense_calibration(db3_path)
        converter.depth_scale = depth_scale
        converter.depth.rotation = R.T.flatten().tolist()
        converter.depth.translation = t.tolist()
        # R_test = R.T
        # t_test = -R.T @ t
        align = partial(align_depth_to_color,
            depth_K=depth_K,
            color_K=color_K,
            color_dist=color_dist,  
            R=R,
            t=-t,
            depth_scale=depth_scale,
            color_shape=(480, 848)
        )
    else:
        align = rs.align(rs.stream.color)
    return pipeline, align, converter, base_dir, tmp_path


def get_frame(pipeline, align, timeout=1500):
    aligned_frames = pipeline.wait_for_frames(timeout)
    if isinstance(align, rs.align):
        aligned_frames = align.process(aligned_frames)
    return aligned_frames


def get_aligned_depth(align, depth_raw):
    if isinstance(align, rs.align):
        return depth_raw
    else: 
        return align(depth_raw)


def load_realsense_calibration(db3_path):
    """
    Extract RealSense calibration directly from a native .db3 file.

    Returns:
        depth_K      : 3x3 depth intrinsic matrix
        color_K      : 3x3 color intrinsic matrix
        color_dist   : 5 distortion coefficients
        R            : depth -> color rotation
        t            : depth -> color translation, meters
        depth_scale  : meters per Z16 unit
    """

    conn = sqlite3.connect(db3_path)

    def get_latest(topic_name):
        row = conn.execute(
            """
            SELECT data
            FROM messages
            WHERE topic_id = (
                SELECT id FROM topics WHERE name = ?
            )
            ORDER BY timestamp DESC
            LIMIT 1
        """,
            (topic_name,),
        ).fetchone()

        if row is None:
            raise RuntimeError(f"Topic not found or empty: {topic_name}")

        data = bytes(row[0])

        # ROS2 CDR serialization of std_msgs/msg/String:
        #
        # bytes 0-3 : CDR encapsulation header
        # bytes 4-7 : uint32 string length
        # bytes 8+  : UTF-8 string
        if len(data) < 8:
            raise RuntimeError(f"Metadata message too short: {len(data)} bytes")

        string_length = int.from_bytes(data[4:8], byteorder="little", signed=False)

        payload = data[8 : 8 + string_length]

        # ROS2 strings include the terminating '\0'
        payload = payload.rstrip(b"\x00")

        return payload.decode("utf-8")

    # ------------------------------------------------------------
    # Depth camera info
    # ------------------------------------------------------------
    depth_info = get_latest("/device_0/sensor_0/Depth_0/camera_info")

    # width=848;height=480;fx=... etc.
    d = {}
    for item in depth_info.split(";"):
        key, value = item.split("=", 1)
        d[key] = value

    fx_d = float(d["fx"])
    fy_d = float(d["fy"])
    cx_d = float(d["ppx"])
    cy_d = float(d["ppy"])

    depth_K = np.array([[fx_d, 0.0, cx_d], [0.0, fy_d, cy_d], [0.0, 0.0, 1.0]], dtype=np.float64)

    # ------------------------------------------------------------
    # Color camera info
    # ------------------------------------------------------------
    color_info = get_latest("/device_0/sensor_1/Color_0/camera_info")

    d = {}
    for item in color_info.split(";"):
        key, value = item.split("=", 1)
        d[key] = value

    fx_c = float(d["fx"])
    fy_c = float(d["fy"])
    cx_c = float(d["ppx"])
    cy_c = float(d["ppy"])

    color_dist = np.array([float(x) for x in d["coeffs"].split(",")], dtype=np.float64)

    color_K = np.array([[fx_c, 0.0, cx_c], [0.0, fy_c, cy_c], [0.0, 0.0, 1.0]], dtype=np.float64)

    # ------------------------------------------------------------
    # Depth -> color extrinsics
    # ------------------------------------------------------------
    depth_tf = get_latest("/device_0/sensor_0/Depth_0/tf/ref_0")

    color_tf = get_latest("/device_0/sensor_1/Color_0/tf/ref_0")

    def parse_tf(s):
        d = {}
        for item in s.split(";"):
            key, value = item.split("=", 1)
            d[key] = value

        R = np.array([float(x) for x in d["rotation"].split(",")], dtype=np.float64).reshape(3, 3)

        t = np.array([float(x) for x in d["translation"].split(",")], dtype=np.float64)

        return R, t

    R_depth, t_depth = parse_tf(depth_tf)
    R_color, t_color = parse_tf(color_tf)

    # Both transforms are relative to ref_0.
    # depth -> ref_0 -> color
    R = R_color.T @ R_depth
    t = R_color.T @ (t_depth - t_color)

    # ------------------------------------------------------------
    # Depth units
    # ------------------------------------------------------------
    depth_units = get_latest("/device_0/sensor_0/option/Depth_Units/value")

    # Depending on recording, this may simply be "0.001"
    # or contain additional fields.
    try:
        depth_scale = float(depth_units)
    except ValueError:
        depth_scale = None

        for item in depth_units.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                if key.lower() in ("value", "depth_units", "depthunits"):
                    depth_scale = float(value)
                    break

        if depth_scale is None:
            raise RuntimeError(f"Could not parse Depth_Units: {depth_units}")

    conn.close()

    return (depth_K, color_K, color_dist, R, t, depth_scale)
