#!/usr/bin/env python3
"""
    This node reads an point cloud from Waymo Open Dataset, converts it to PCL and publishd to a determined topic
"""
import os
import sys
import numpy as np
import pathlib
from matplotlib import cm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *


def get_pointcloud(frame):
    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
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

    return points, points_cp


def get_camera_image(frame):
    # Gets and orders camera images from left to right
    cameras_images = {}
    for i, image in enumerate(frame.images):
        decoded_image = get_frame_image(image)
        cameras_images[image.name] = decoded_image[...,::-1]
    # # Create a list of tuples (key, image_data)
    # key_image_tuples = [(key, cameras_images[key]) for key in cameras_images]
    # # Sort the list of tuples based on the custom order
    # sorted_tuples = sorted(key_image_tuples, key=lambda x: self.cams_order.index(x[0]))
    # # Extract image data from sorted tuples
    # ordered_images = [item[1] for item in sorted_tuples]
    ordered_images = [cameras_images[1]]

    return ordered_images

def cart2hom( pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def project_pointcloud_to_camera(points_3d, V2C):
    # Convert pointcloud in cartesian to pointcloud in homogeneus coordinates
    points_3d = cart2hom(points_3d)  # nx4
    pts_3d_cam =  np.dot(points_3d, np.transpose(V2C))

    return pts_3d_cam
    # return self.project_ref_to_rect(pts_3d_camid)#apply R0

def project_cam3d_to_image(pts_3d_rect, P):
    """ Input: nx3 points in rect camera coord.
        Output: nx2 points in image coord.
    """
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def plot_referenced_pointcloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame])

if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_valid_scene")
    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    ##############################################################
    ## Iterate through Dataset Scenes
    ##############################################################
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        scene_name = scene_path.stem
        print(scene_name)
        
        frame = next(load_frame(scene_path))

        # Camera parameters
        intrinsic_params = np.array((frame.context.camera_calibrations[0].intrinsic), dtype=np.float32)
        intrinsic_matrix = [[intrinsic_params[0], 0, intrinsic_params[2]],
                            [0, intrinsic_params[1], intrinsic_params[3]],
                            [0, 0, 1]]
        distortion = np.array([intrinsic_params[4], intrinsic_params[5], intrinsic_params[6], intrinsic_params[7], intrinsic_params[8]])

        extrinsic_matrix = np.array((frame.context.camera_calibrations[0].extrinsic.transform), dtype=np.float32).reshape(4, 4)
        # Extract rotation and translation components from the extrinsic matrix
        rotation = extrinsic_matrix[:3, :3]
        translation = extrinsic_matrix[:3, 3]
        # Invert the rotation matrix
        rotation_inv = np.linalg.inv(rotation)
        # Invert the translation
        translation_inv = -np.dot(rotation_inv, translation)
        # Construct the new extrinsic matrix (from camera to vehicle)
        extrinsic_matrix_inv = np.zeros((4, 4), dtype=np.float32)
        extrinsic_matrix_inv[:3, :3] = rotation_inv
        extrinsic_matrix_inv[:3, 3] = translation_inv
        extrinsic_matrix_inv[3, 3] = 1.0
        extrinsic_matrix_inv = extrinsic_matrix_inv[:3,:]

        # Frame front image
        front_image = get_camera_image(frame)[0]
        image_height, image_width, _ = front_image.shape

        # Frame pointcloud
        point_cloud, points_cp = get_pointcloud(frame)
        print("PointCloud: ", point_cloud)
        # plot_referenced_pointcloud(point_cloud)

        ########################################################################################
        ## Project Velodyne points to Camera
        ########################################################################################
        # Project 3D points in vehicle reference to 3D points in camera reference using extrinsic params
        point_cloud_hom = cart2hom(point_cloud)  # nx4
        point_cloud_cam =  np.dot(point_cloud_hom, np.transpose(extrinsic_matrix_inv))
        # plot_referenced_pointcloud(point_cloud_cam)

        ########################################################################################
        ## Rotate pointcloud to match pin hole axis
        ########################################################################################
        ## Rotate point cloud to match pinhole model axis (Z pointing to the front of the car, Y pointing down ans X pointing to the right)
        theta_x = np.pi/2
        theta_y = -np.pi/2
        theta_z = 0
        # Define rotation matrices around X, Y, and Z axes
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
        # Rotate the point cloud 90 degrees in X
        point_cloud_cam = np.dot(Rx, point_cloud_cam.T).T
        # Rotate the point cloud -90 degrees in Y
        point_cloud_cam = np.dot(Ry, point_cloud_cam.T).T

        # plot_referenced_pointcloud(point_cloud_cam)

        ########################################################################################
        ## Remove points with z<0
        ########################################################################################
        # We only have to project in the image the points that are in front of the car > remove points with z<0
        point_cloud_cam = point_cloud_cam[point_cloud_cam[:, 2] >= 0]

        # Plot pointcloud referenced to the camera
        # plot_referenced_pointcloud(point_cloud_cam)

        # Get a color map based on the Z depth
        # Normalize Z coordinate values
        min_depth = np.min(point_cloud_cam[:, 2])
        max_depth = np.max(point_cloud_cam[:, 2])
        normalized_depth = (point_cloud_cam[:, 2] - min_depth) / (max_depth - min_depth)
        cmap = cm.get_cmap('jet')  # You can choose any colormap you prefer
        colors = cmap(normalized_depth)

        ########################################################################################
        ## Project 3D points in camera reference in 2D points in image reference using intrinsic params
        ########################################################################################
        point_cloud_cam_hom = cart2hom(point_cloud_cam)
        # Compute perspective projection matrix
        P = np.hstack((intrinsic_matrix, np.zeros((3, 1))))
        point_cloud_image = np.dot(P, point_cloud_cam_hom.T).T
        point_cloud_image = point_cloud_image[:, :2] / point_cloud_image[:, 2:]

        # Filtered 2D points > remove points out of the image FOV
        filtered_indices = (
            (point_cloud_image[:, 0] >= 0) & (point_cloud_image[:, 0] < image_width) &
            (point_cloud_image[:, 1] >= 0) & (point_cloud_image[:, 1] < image_height)
        )
        # point_cloud_image = point_cloud_image[
        #     (point_cloud_image[:, 0] >= 0) & (point_cloud_image[:, 0] < image_width) &
        #     (point_cloud_image[:, 1] >= 0) & (point_cloud_image[:, 1] < image_height)
        # ]
        point_cloud_image = point_cloud_image[filtered_indices]
        colors = colors[filtered_indices]

        # Plot image
        plt.imshow(cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB))
        plt.scatter(point_cloud_image[:, 0], point_cloud_image[:, 1], color=colors, s=2)
        plt.colorbar(label='Depth')
        plt.show()

