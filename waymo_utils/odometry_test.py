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
        """
        Get incremental pose of the vehicle
        """
        transform_matrix = np.array(frame.pose.transform).reshape(4, 4)
        if prev_transform_matrix is None:
            inc_odom = [0, 0, 0, 0, 0, 0]  # Initialize Euler angles to 0
        else:
            prev_pose = get_pose(prev_transform_matrix)
            pose = get_pose(transform_matrix)

            delta_position = np.subtract(pose[:3], prev_pose[:3])
            delta_orientation = np.subtract(pose[3:], prev_pose[3:])  # Calculate delta Euler angles

            # Normalize the delta_orientation
            delta_orientation = [normalize_angle(angle) for angle in delta_orientation]
            inc_odom = np.concatenate((delta_position, delta_orientation))
        
        return inc_odom, transform_matrix

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
axis_length = 0.5  # Set the length of each axis for visualization

transform_matrix = None
for scene_index, scene_path in enumerate(tfrecord_list):
    scene_name = scene_path.stem
    frame_n = 0  # Scene frames counter
    prev_point = None

    ## Evaluation > position and orientation vectors to store vehicles poses
    initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64) # Quaternion initialized to no rotation
    accumulated_pose = np.copy(initial_pose)
    for frame in load_frame(scene_path):
        if frame is None:
            continue

        """
        REMOVE
        """
        transform_matrix = np.array(frame.pose.transform).reshape(4, 4)
        pose = get_pose(transform_matrix)
        print(pose)

        """
        VISUALIZATION
        """
        # Store the position as a 3D point
        current_point = np.array([pose[0], pose[1], pose[2]])
        points.append(current_point)

        # Create an axis for the pose
        R_matrix = transform_matrix[:3, :3]
        pose_translation = current_point
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=pose_translation)
        
        # Apply the rotation from the pose to the axis frame
        axis.rotate(R_matrix, center=pose_translation)
        vis.add_geometry(axis)  # Add axis to the visualization

        # Connect the current point to the previous one to show the path
        if prev_point is not None:
            lines.append([len(points) - 2, len(points) - 1])

        prev_point = current_point  # Update previous point

        # # Obtain odometry increment (in frame 0 will be 0)
        # inc_odom, transform_matrix = process_odometry(frame, transform_matrix)
        # increment_pose = np.array(inc_odom)

        # if frame_n == 0:
        #     accumulated_pose = increment_pose
        # else:
        #     inc_position = increment_pose[:3]
        #     accumulated_position = accumulated_pose[:3] + inc_position
        #     inc_euler = increment_pose[3:]
        #     acc_euler = accumulated_pose[3:] + inc_euler  # Accumulate Euler angles
        #     # Normalize the accumulated Euler angles
        #     acc_euler = [normalize_angle(angle) for angle in acc_euler]
        #     accumulated_pose = np.concatenate((accumulated_position, acc_euler))
        frame_n += 1

        # print("Accumulated Vehicle Pose (incremental movement): \n")
        # print("Frame: \n")
        # print("X: ",         accumulated_pose[0])
        # print("Y: ",         accumulated_pose[1])
        # print("Z: ",         accumulated_pose[2])
        # print("Roll: ",      accumulated_pose[3])
        # print("Pitch: ",     accumulated_pose[4])
        # print("Yaw: ",       accumulated_pose[5])

    # Convert points and lines to Open3D format
    points_o3d = o3d.utility.Vector3dVector(points)
    line_set.points = points_o3d
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Add the LineSet to the visualization
    line_set.paint_uniform_color([0, 0, 1])  # Optional: make lines blue for visibility
    vis.add_geometry(line_set)

    # Run the visualization
    vis.run()
    vis.destroy_window()