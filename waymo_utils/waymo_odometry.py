import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)
print(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *


def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def ominus(T1, T2):
    """Compute the relative transformation (T1 ominus T2)."""
    return np.linalg.inv(T2) @ T1

def get_pose(T):
    position = T[:3, 3]
    position_x = position[0]
    position_y = position[1]
    position_z = position[2]

    R_matrix = T[:3, :3]
    rotation = R.from_matrix(R_matrix)
    quaternion = rotation.as_quat()

    orientation_x = quaternion[0]
    orientation_y = quaternion[1]
    orientation_z = quaternion[2]
    orientation_w = quaternion[3]

    roll, pitch, yaw = quaternion_to_euler(orientation_w, 
                                           orientation_x, 
                                           orientation_y, 
                                           orientation_z)
    
    return [position_x, position_y, position_z, roll, pitch, yaw, R_matrix]


if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_test_scene")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    prev_transform_matrix = None
    incremental_positions = []
    poses = []
    orientations = []

    # for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
    scene_path = tfrecord_list[0]
    frame_i = 0
    for frame in load_frame(scene_path):
        print(frame.context.name)

        point_cloud, points_cp = get_pointcloud(frame)
        vehicle_pose_vector = frame.pose.transform
        transform_matrix = np.array(vehicle_pose_vector).reshape(4, 4)
        pose = get_pose(transform_matrix)
        print("X: ",        pose[0])
        print("Y: ",        pose[1])
        print("Z: ",        pose[2])
        print("Roll: ",     pose[3])
        print("Pitch: ",    pose[4])
        print("Yaw: ",      pose[5])
        poses.append(pose[:3])


        if prev_transform_matrix is not None:
            inc_odom_matrix = ominus(transform_matrix, prev_transform_matrix)

            inc_odom = get_pose(inc_odom_matrix)
            # print("X: ",        inc_odom[0])
            # print("Y: ",        inc_odom[1])
            # print("Z: ",        inc_odom[2])
            # print("Roll: ",     inc_odom[3])
            # print("Pitch: ",    inc_odom[4])
            # print("Yaw: ",      inc_odom[5])
            incremental_positions.append(inc_odom[:3])  # Only x, y, z
            orientations.append(inc_odom[6])  # Rotation matrix

        prev_transform_matrix = transform_matrix
        frame_i += 1
        if (frame_i == 20):
            break

# Plot the incremental positions and orientations
if incremental_positions:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    incremental_positions = np.array(incremental_positions)

    # Plot positions
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], 'o-', label='Position')

    # Plot orientations as arrows
    for pos, ori in zip(poses, orientations):
        ax.quiver(pos[0], pos[1], pos[2], ori[0, 0], ori[1, 0], ori[2, 0], color='r', length=0.5, normalize=True)
        ax.quiver(pos[0], pos[1], pos[2], ori[0, 1], ori[1, 1], ori[2, 1], color='g', length=0.5, normalize=True)
        ax.quiver(pos[0], pos[1], pos[2], ori[0, 2], ori[1, 2], ori[2, 2], color='b', length=0.5, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Incremental Positions and Orientations')
    ax.grid(True)
    ax.legend()
    plt.show()
