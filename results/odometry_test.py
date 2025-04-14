import os
import sys
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_open_dataset.utils import frame_utils  # Adjust import if necessary

def get_pose(T):
    position = T[:3, 3]
    R_matrix = T[:3, :3]
    rotation = R.from_matrix(R_matrix)
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
    return [position[0], position[1], position[2], roll, pitch, yaw]

def normalize_angle(angle):
    """ Normalize the angle to be within the range [-\u03c0, \u03c0] """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def process_odometry(frame, position_noise_std=0.05, orientation_noise_std=0.01, initial_transform_matrix=None):
    """
    Get incremental pose of the vehicle with constant Gaussian noise.
    """
    transform_matrix = np.array(frame.pose.transform).reshape(4, 4)
    if initial_transform_matrix is None:
        initial_transform_matrix = transform_matrix

    relative_transform = np.linalg.inv(initial_transform_matrix) @ transform_matrix
    relative_pose = get_pose(relative_transform)
    position_noise = [
        np.random.normal(0, position_noise_std) if i < 2 else np.random.normal(0, position_noise_std * 0.1)
        for i in range(3)
    ]  # Reduce Z noise significantly
    orientation_noise = [
        np.random.normal(0, orientation_noise_std) for _ in range(3)
    ]

    # Aplicar ruido directamente a la pose relativa
    noisy_position = [
        relative_pose[i] + position_noise[i] for i in range(3)
    ]
    noisy_orientation = [
        relative_pose[i+3] + orientation_noise[i] for i in range(3)
    ]

    # Generar la pose odométrica con ruido
    odometry_pose = noisy_position + noisy_orientation

    return relative_pose, odometry_pose, transform_matrix, initial_transform_matrix

def concatenate_pcd_returns(pcd_return_1, pcd_return_2):
    points, points_cp = pcd_return_1
    points_ri2, points_cp_ri2 = pcd_return_2
    points_concat = np.concatenate(points + points_ri2, axis=0)
    points_cp_concat = np.concatenate(points_cp + points_cp_ri2, axis=0)
    return points_concat, points_cp_concat

scene = "individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels"
scene_path = os.path.join(src_dir, "dataset/final_tests_scene/" + scene + ".tfrecord")
points = []
noisy_points = []
lines = []
noisy_lines = []
line_set = o3d.geometry.LineSet()
noisy_line_set = o3d.geometry.LineSet()

vis = o3d.visualization.Visualizer()
vis.create_window()

frame_n = 0
prev_point = None
prev_noisy_point = None
initial_transform_matrix = None
position_noise_cumulative = [0, 0, 0]
orientation_noise_cumulative = [0, 0, 0]

## Draw map landmarks
signs_map_file = os.path.join(src_dir, "dataset/pointclouds/pointcloud_concatenated" + scene +".csv")
signs_map = pd.read_csv(signs_map_file)
map_points = signs_map.iloc[:, :3].values
map_point_cloud = o3d.geometry.PointCloud()
map_point_cloud.points = o3d.utility.Vector3dVector(map_points)
color = np.array([[1.0, 0.0, 0.0]] * len(map_points))
map_point_cloud.colors = o3d.utility.Vector3dVector(color)
vis.add_geometry(map_point_cloud)

distance = 0

for frame in load_frame(scene_path):
    if frame is None:
        continue

    relative_pose, noisy_relative_pose, transform_matrix, initial_transform_matrix = process_odometry(frame, position_noise_std=0.05, orientation_noise_std=0.01, initial_transform_matrix=initial_transform_matrix)

    point = [relative_pose[0], relative_pose[1], relative_pose[2]]
    points.append(point)

    noisy_point = [noisy_relative_pose[0], noisy_relative_pose[1], noisy_relative_pose[2]]
    noisy_points.append(noisy_point)

    incremental_distance = np.linalg.norm(np.array(point) - np.array(prev_point)) if prev_point is not None else 0
    distance += incremental_distance

    if prev_point is not None:
        lines.append([len(points) - 2, len(points) - 1])
    if prev_noisy_point is not None:
        noisy_lines.append([len(noisy_points) - 2, len(noisy_points) - 1])

    prev_point = point
    prev_noisy_point = noisy_point
    frame_n += 1

    if frame_n != 1:
        continue

    range_images, camera_projections, segmentation_labels, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)

    def _range_image_to_pcd(ri_index=0):
        points, points_cp = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=ri_index)
        return points, points_cp

    points_return1 = _range_image_to_pcd()
    points_return2 = _range_image_to_pcd(1)

    pointcloud, points_cp = concatenate_pcd_returns(points_return1, points_return2)
    transform = np.reshape(np.array(frame.pose.transform), [4, 4])
    points_homogeneous = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    global_points = np.dot(transform, points_homogeneous.T).T
    global_points = global_points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 1.0

points_o3d = o3d.utility.Vector3dVector(points)
line_set.points = points_o3d
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.paint_uniform_color([0, 0, 1])
vis.add_geometry(line_set)

noisy_line_set.points = o3d.utility.Vector3dVector(noisy_points)
noisy_line_set.lines = o3d.utility.Vector2iVector(noisy_lines)
noisy_line_set.paint_uniform_color([1, 0, 0])
vis.add_geometry(noisy_line_set)

vis.run()
vis.clear_geometries()
vis.destroy_window()

print(frame_n)
print("Distance traveled by the car: ", distance)