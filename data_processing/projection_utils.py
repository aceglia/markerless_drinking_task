import open3d as o3d
import numpy as np
import cv2

def get_reduced_pc(depth, color, crops_pos, limit_depth, camera_config):
    mask = np.zeros(depth.shape[:2], dtype=np.uint8)
    points = np.array(crops_pos, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    if color is not None:
        color = cv2.bitwise_and(color, color, mask=mask)
    depth = cv2.bitwise_and(depth, depth, mask=mask)
    depth = np.where((depth > limit_depth[0]) & (depth < limit_depth[1]), depth, 0)
    pc = pc_from_rgbd(depth, color, camera_config)
    return pc

def crop_pc(pc, crops_pos, limit_depth, camera_config):
    z_min, z_max = np.array(limit_depth) * camera_config.depth_scale
    x_min, y_min = np.min(np.array(crops_pos), axis=0)
    x_max, y_max = np.max(crops_pos, axis=0)
    fx=camera_config.depth.fx,
    fy=camera_config.depth.fy,
    cx=camera_config.depth.ppx,
    cy=camera_config.depth.ppy,
    x_min = (x_min - cx) * z_min / fx
    x_max = (x_max - cx) * z_max / fx
    y_min = (y_min - cy) * z_min / fy
    y_max = (y_max - cy) * z_max / fy
    bbox_min = np.array([x_min, y_min, z_min])
    bbox_max = np.array([x_max, y_max, z_max])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    bounding_box.color = (1, 0, 0)
    return pc.crop(bounding_box)


def get_3d_bbox(pixels_pos, depth, camera_config):
    z_min, z_max = np.array(depth) * camera_config.depth_scale
    x_min, y_min = np.min(np.array(pixels_pos), axis=0)
    x_max, y_max = np.max(pixels_pos, axis=0)
    fx=camera_config.depth.fx,
    fy=camera_config.depth.fy,
    cx=camera_config.depth.ppx,
    cy=camera_config.depth.ppy,
    z = depth[pixels_pos[1], pixels_pos[0]]
    x_min = (x_min - cx) * z_min / fx
    x_max = (x_max - cx) * z_max / fx
    y_min = (y_min - cy) * z_min / fy
    y_max = (y_max - cy) * z_max / fy
    return np.array([x, y, z])


def pc_from_rgbd(depth, color, camera_config):
    depth_o3d = o3d.geometry.Image(depth)
    color_o3d = o3d.geometry.Image(color)
    intrincics = o3d.camera.PinholeCameraIntrinsic(
        width=camera_config.depth.width,
        height=camera_config.depth.height,
        fx=camera_config.depth.fx,
        fy=camera_config.depth.fy,
        cx=camera_config.depth.ppx,
        cy=camera_config.depth.ppy,
    )
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, convert_rgb_to_intensity=False
    )
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrincics)
    return o3d_pcd


def get_mid_point(should_pos, distance_factor=0.8):
    distance = np.linalg.norm(
        [(should_pos[1, 0] - should_pos[0, 0]), (should_pos[1, 1] - should_pos[0, 1])]
    ) * distance_factor
    midline = [(should_pos[1, 0] + should_pos[0, 0]) / 2, (should_pos[1, 1] + should_pos[0, 1]) / 2]
    unit_vector = get_vec_from_should(should_pos)
    mid_point = midline + distance * unit_vector
    return mid_point, unit_vector


def get_vec_from_should(should_pos):
    vect = [should_pos[1, 0] - should_pos[0, 0], should_pos[1, 1] - should_pos[0, 1]]
    perp_vector = [-vect[1], vect[0]]
    norm = np.linalg.norm(perp_vector)
    unit_vector = perp_vector / norm
    return unit_vector


def perform_icp(ref_pc, target_pc, spheres, threshold=0.1, initial_guess=np.eye(4), show=False):
    ref_pc = ref_pc.voxel_down_sample(0.01)
    target_pc = target_pc.voxel_down_sample(0.01)
    ref_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    target_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    ref_pc.orient_normals_towards_camera_location([0, 0, 0])
    target_pc.orient_normals_towards_camera_location([0, 0, 0])

    loss = o3d.pipelines.registration.TukeyLoss(k=0.02)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        ref_pc,
        target_pc,
        threshold,
        initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
    )

    new_spheres = []
    if not isinstance(spheres, list):
        spheres = [spheres]
    for sphere in spheres:
        new_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
        center_trans = (reg_p2p.transformation @ np.append(sphere.get_center(), 1.0))[:3]
        new_sphere.translate(center_trans)
        new_sphere.paint_uniform_color([1, 0, 0])
        new_spheres.append(new_sphere)
    if show:
        o3d_pcd_result = o3d.geometry.PointCloud(ref_pc)
        o3d_pcd_result.transform(reg_p2p.transformation)
        o3d_pcd_result.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries(
            [
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]),
                target_pc,
                o3d_pcd_result,
                *new_spheres,
            ]
        )
    if len(new_spheres) == 1:
        new_spheres = new_spheres[0]
    return new_spheres, reg_p2p.transformation
