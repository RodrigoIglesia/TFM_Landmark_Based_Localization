import os
import sys
import pathlib
import numpy as np
import open3d as o3d
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

def process_odometry(frame, prev_transform_matrix):
    transform_matrix = np.array(frame.pose.transform).reshape(4, 4)
    if prev_transform_matrix is None:
        inc_odom = [0, 0, 0, 0, 0, 0]
    else:
        prev_pose = get_pose(prev_transform_matrix)
        pose = get_pose(transform_matrix)
        delta_position = np.subtract(pose[:3], prev_pose[:3])
        delta_orientation = np.subtract(pose[3:], prev_pose[3:])
        delta_orientation = [normalize_angle(angle) for angle in delta_orientation]
        inc_odom = np.concatenate((delta_position, delta_orientation))
    return inc_odom, transform_matrix

def concatenate_pcd_returns(pcd_return_1, pcd_return_2):
    points, points_cp = pcd_return_1
    points_ri2, points_cp_ri2 = pcd_return_2
    points_concat = np.concatenate(points + points_ri2, axis=0)
    points_cp_concat = np.concatenate(points_cp + points_cp_ri2, axis=0)
    print(f'points_concat shape: {points_concat.shape}')
    print(f'points_cp_concat shape: {points_cp_concat.shape}')
    return points_concat, points_cp_concat

# Read dataset
dataset_path = os.path.join(src_dir, "dataset/waymo_test_scene")
tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

# Prepare Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Prepare Open3D geometry objects
points = []
lines = []
line_set = o3d.geometry.LineSet()
axis_length = 0.005  # Axis length for visualization

transform_matrix = None
for scene_index, scene_path in enumerate(tfrecord_list):
    scene_name = scene_path.stem
    print("SCENE: ", scene_name)
    frame_n = 0  # Scene frames counter
    prev_point = None

    initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    prev_transform_matrix = None
    accumulated_pose = np.copy(initial_pose)
    for frame in load_frame(scene_path):
        if frame is None:
            continue

        incremental_pose, transform_matrix = process_odometry(frame, prev_transform_matrix)
        print(incremental_pose)

        point = [incremental_pose[0], incremental_pose[1], incremental_pose[2]]
        points.append(point)

        # Create an axis for the pose
        R_matrix = transform_matrix[:3, :3]
        pose_translation = point
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=pose_translation)
        axis.rotate(R_matrix, center=pose_translation)
        vis.add_geometry(axis)

        # Connect the current point to the previous one to show the path
        if prev_point is not None:
            lines.append([len(points) - 2, len(points) - 1])

        prev_point = point
        prev_transform_matrix = transform_matrix
        frame_n += 1

        # Obtain the point cloud for the first frame
        if frame_n == 1:
            # Parse range image and convert to point cloud
            range_images, camera_projections, segmentation_labels, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)

            # Get points labeled for first and second return
            def _range_image_to_pcd(ri_index=0):
                points, points_cp = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose, ri_index=ri_index)
                return points, points_cp

            # First and second lidar returns
            points_return1 = _range_image_to_pcd()
            points_return2 = _range_image_to_pcd(1)

            # Combine both returns for a denser point cloud
            pointcloud, points_cp = concatenate_pcd_returns(points_return1, points_return2)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Optional: set color for visibility

            # Add point cloud to the visualizer
            vis.add_geometry(pcd)

    # Convert points and lines to Open3D format
    points_o3d = o3d.utility.Vector3dVector(points)
    line_set.points = points_o3d
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 0, 1])  # Optional: set color for path lines
    vis.add_geometry(line_set)

    # Run the visualization
    vis.run()
    vis.destroy_window()
