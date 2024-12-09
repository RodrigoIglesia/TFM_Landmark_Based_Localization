
"""
Utils functions to operate with poses and frames transformations
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


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


def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-norm quaternion")
    return q / norm


def quaternion_multiply(q1, q2):
    """ Multiplies two quaternions in the format [x, y, z, w]. """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]


def quaternion_conjugate(q):
    """ Returns the conjugate (inverse) of a quaternion in the format [x, y, z, w]. """
    x, y, z, w = q
    return [-x, -y, -z, w]


def quaternion_difference(q1, q2):
    """ Returns the quaternion representing the rotation from q1 to q2 in the format [x, y, z, w]. """
    q1_inv = quaternion_conjugate(q1)
    return quaternion_multiply(q1_inv, q2)


def create_pose_frame(pose, size=0.6):
    x, y, z, roll, pitch, yaw = pose
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[x, y, z])
    # Create rotation matrix from roll, pitch, yaw
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rot_matrix = r.as_matrix()
    mesh_frame.rotate(rot_matrix, center=[x, y, z])
    return mesh_frame

def get_pose(T):
    position = T[:3, 3]
    R_matrix = T[:3, :3]
    rotation = R.from_matrix(R_matrix)
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)

    return [position[0], position[1], position[2], roll, pitch, yaw]


def normalize_angle(angle):
    """ Normalize the angle to be within the range [-π, π] """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def create_homogeneous_matrix(position):
    """
    Create a 4x4 homogeneous transformation matrix from a position vector.

    Parameters:
        position (list or np.ndarray): Vector with 6 values [x, y, z, roll, pitch, yaw].

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
    """
    if len(position) != 6:
        raise ValueError("Position vector must have exactly 6 elements.")
    
    x, y, z = position[:3]
    roll, pitch, yaw = position[3:]

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Homogeneous transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = [x, y, z]

    return transform

def comp_poses(accumulated_pose, increment):
    """
    Accumulate poses by multiplying their homogeneous transformation matrices.

    Parameters:
        accumulated_pose (list or np.ndarray): Current pose vector [x, y, z, roll, pitch, yaw].
        increment (list or np.ndarray): Incremental pose vector [x, y, z, roll, pitch, yaw].

    Returns:
        np.ndarray: New accumulated pose vector [x, y, z, roll, pitch, yaw].
    """
    # Create homogeneous matrices
    accumulated_matrix = create_homogeneous_matrix(accumulated_pose)
    increment_matrix = create_homogeneous_matrix(increment)

    # Multiply matrices
    result_matrix = accumulated_matrix @ increment_matrix

    # Extract translation
    x, y, z = result_matrix[:3, 3]

    # Extract rotation (roll, pitch, yaw)
    roll = np.arctan2(result_matrix[2, 1], result_matrix[2, 2])
    pitch = np.arcsin(-result_matrix[2, 0])
    yaw = np.arctan2(result_matrix[1, 0], result_matrix[0, 0])

    return np.array([x, y, z, roll, pitch, yaw])

def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom
