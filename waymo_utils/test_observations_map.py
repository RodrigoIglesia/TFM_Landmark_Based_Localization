import sys
import os
import csv
from WaymoParser import *
from waymo_3d_parser import *
import numpy as np

# Function to read poses from a CSV file without a header
def read_csv_no_header(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        return [[float(value) for value in row] for row in reader]

# Function to add pose axes as points to the point cloud
def add_poses_to_pointcloud(points, poses, axis_size=0.5):
    axes_points = []
    
    for pose in poses:
        if len(pose) < 2:
            raise ValueError("Each row in the CSV file must have at least two values (X, Y).")
        x, y = pose[0], pose[1]
        z = 0  # Assuming Z = 0 for the poses
        
        # Define small "axes" points around the pose
        axes_points.extend([
            [x, y, z],  # Origin
            [x + axis_size, y, z],  # X-axis
            [x, y + axis_size, z],  # Y-axis
            [x, y, z + axis_size],  # Z-axis
        ])
    
    # Combine original points and pose axes points
    combined_points = np.vstack((points, np.array(axes_points)))
    return combined_points

if __name__ == "__main__":
    # Add project root to Python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/final_tests_scene")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        frame = next(load_frame(scene_path)) # Get only first frame
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        frame.lasers.sort(key=lambda laser: laser.name)

        # Get points labeled for first and second return
        # Parse range image for lidar 1
        def _range_image_to_pcd(ri_index = 0):
            points, points_cp = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose,
                ri_index=ri_index)
            return points, points_cp
        
        # Return of the first 2 lidar scans
        points_return1 = _range_image_to_pcd()
        points_return2 = _range_image_to_pcd(1)

        points, points_cp = concatenate_pcd_returns(points_return1, points_return2)

        # Read poses from the CSV file
        print(src_dir)
        # csv_file_path = os.path.join(src_dir, "/pointcloud_clustering/map/signs_map_features_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")  # Replace with the path to your CSV
        poses = read_csv_no_header("/home/rodrigo/catkin_ws/src/TFM_Landmark_Based_Localization/pointcloud_clustering/map/signs_map_features_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")

        # Add poses to the point cloud
        points_with_poses = add_poses_to_pointcloud(points, poses)

        # Plot the updated point cloud
        plot_referenced_pointcloud(points_with_poses, True)
