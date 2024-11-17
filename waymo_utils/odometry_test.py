import os
import sys
import pathlib
import numpy as np
import open3d as o3d
import csv  # Necesario para leer el CSV
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
    """ Normalize the angle to be within the range [-π, π] """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def process_odometry(frame, position_noise_cumulative, orientation_noise_cumulative, initial_transform_matrix=None, position_noise_std=0.1, orientation_noise_std=0.01):
    """
    Get incremental pose of the vehicle with cumulative noise.
    """
    # Extract the transform matrix for the current frame
    transform_matrix = np.array(frame.pose.transform).reshape(4, 4)
    
    # Initialize the initial frame as the origin if not already done
    if initial_transform_matrix is None:
        initial_transform_matrix = transform_matrix
    
    # Compute the relative pose: relative_pose = initial_transform_matrix^-1 * transform_matrix
    relative_transform = np.linalg.inv(initial_transform_matrix) @ transform_matrix
    relative_pose = get_pose(relative_transform)
    print(f"Vehicle relative pose: {relative_pose}")

    # Generar ruido gaussiano y acumularlo en las variables de ruido acumulado
    position_noise_cumulative = [
        position_noise_cumulative[i] + np.random.normal(0, position_noise_std)
        for i in range(3)
    ]
    orientation_noise_cumulative = [
        orientation_noise_cumulative[i] + np.random.normal(0, orientation_noise_std)
        for i in range(3)
    ]

    # Añadir el ruido acumulado a la pose relativa para obtener la odometría con ruido
    noisy_position = [
        relative_pose[i] + position_noise_cumulative[i] for i in range(3)
    ]
    noisy_orientation = [
        relative_pose[i+3] + orientation_noise_cumulative[i] for i in range(3)
    ]

    # Actualizar la odometría con los valores ruidosos acumulativos
    odometry_pose = noisy_position + noisy_orientation
    print(f"Vehicle odometry Pose (with cumulative noise): {odometry_pose}")
    
    return relative_pose, odometry_pose, transform_matrix, initial_transform_matrix

def concatenate_pcd_returns(pcd_return_1, pcd_return_2):
    points, points_cp = pcd_return_1
    points_ri2, points_cp_ri2 = pcd_return_2
    points_concat = np.concatenate(points + points_ri2, axis=0)
    points_cp_concat = np.concatenate(points_cp + points_cp_ri2, axis=0)
    return points_concat, points_cp_concat

# Ruta del CSV con los elementos verticales
csv_path = os.path.join(src_dir, "pointcloud_clustering/map/signs_map_features_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")

# Leer los datos del CSV sin usar pandas
vertical_elements = []
with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        # Agregar las coordenadas z, roll, pitch y yaw constantes
        vertical_elements.append([x, y, -1.8, 0.0, 0.0, 0.0])

# Read dataset
dataset_path = os.path.join(src_dir, "dataset/final_tests_scene")
tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

# Prepare Open3D geometry objects
points = []
noisy_points = []
lines = []
noisy_lines = []
line_set = o3d.geometry.LineSet()
noisy_line_set = o3d.geometry.LineSet()
axis_length = 0.05  # Axis length for visualization

for scene_index, scene_path in enumerate(tfrecord_list):
    # Prepare Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    scene_name = scene_path.stem
    print("SCENE: ", scene_name)

    frame_n = 0  # Scene frames counter
    prev_point = None
    prev_noisy_point = None
    initial_transform_matrix = None
    position_noise_cumulative = [0, 0, 0]
    orientation_noise_cumulative = [0, 0, 0]

    for frame in load_frame(scene_path):
        if frame is None:
            continue

        # relative_pose, noisy_relative_pose, transform_matrix, initial_transform_matrix = process_odometry(frame, position_noise_cumulative, orientation_noise_cumulative, initial_transform_matrix)
        # print(relative_pose)

        # point = [relative_pose[0], relative_pose[1], relative_pose[2]]
        # points.append(point)

        # noisy_point = [noisy_relative_pose[0], noisy_relative_pose[1], noisy_relative_pose[2]]
        # noisy_points.append(noisy_point)

        # # Create an axis for the pose
        # R_matrix = transform_matrix[:3, :3]
        # pose_translation = point
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=pose_translation)
        # axis.rotate(R_matrix, center=pose_translation)
        # vis.add_geometry(axis)

        # # Connect the current point to the previous one to show the path
        # if prev_point is not None:
        #     lines.append([len(points) - 2, len(points) - 1])
        # if prev_noisy_point is not None:
        #     noisy_lines.append([len(noisy_points) - 2, len(noisy_points) - 1])

        # prev_point = point
        # prev_noisy_point = noisy_point
        frame_n += 1

        # Obtener la nube de puntos para el primer frame
        if frame_n == 1:
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
            pcd.points = o3d.utility.Vector3dVector(global_points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.point_size = 1.0

    # # Convert points and lines to Open3D format
    # points_o3d = o3d.utility.Vector3dVector(points)
    # line_set.points = points_o3d
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.paint_uniform_color([0, 0, 1])  # Optional: set color for path lines
    # vis.add_geometry(line_set)

    # noisy_line_set.points = o3d.utility.Vector3dVector(noisy_points)
    # noisy_line_set.lines = o3d.utility.Vector2iVector(noisy_lines)
    # noisy_line_set.paint_uniform_color([1, 0, 0])  # Red for noisy path
    # vis.add_geometry(noisy_line_set)

    # Añadir ejes de coordenadas para cada elemento vertical
    for element in vertical_elements:
        element_position = element[:3]  # x, y, z
        axis_element = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=element_position)
        vis.add_geometry(axis_element)

    # Run the visualization
    vis.run()
    vis.clear_geometries()
    vis.destroy_window()
