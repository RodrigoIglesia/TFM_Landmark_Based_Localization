"""
Waymo Open Dataset - 3D Semantic Segmentation parser (LiDAR)
Este módulo contiene funciones para:
Leer cada frame del dataset y extraer las nubes de puntos
Representar las nubes de puntos en visualización 3D
"""

import os
import sys

import pathlib
import tensorflow as tf
import numpy as np
import open3d as o3d

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils

from WaymoParser import *

def visualize_pointcloud_return(frame, pcd_return):
    points, points_cp = pcd_return
    points_all = np.concatenate(points, axis=0)
    print(f'points_all shape: {points_all.shape}')
 
    # camera projection corresponding to each point
    points_cp_all = np.concatenate(points_cp, axis=0)
    print(f'points_cp_all shape: {points_cp_all.shape}')
 
    show_point_cloud(points_all)


def show_point_cloud(points):
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    pcd.points = o3d.utility.Vector3dVector(points)

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    vis.run()


def concatenate_pcd_returns(pcd_return_1, pcd_return_2):
    points, points_cp = pcd_return_1
    points_ri2, points_cp_ri2 = pcd_return_2
    points_concat = np.concatenate(points + points_ri2, axis=0)
    points_cp_concat = np.concatenate(points_cp + points_cp_ri2, axis=0)
    print(f'points_concat shape: {points_concat.shape}')
    print(f'points_cp_concat shape: {points_cp_concat.shape}')
    return points_concat, points_cp_concat


if __name__ == "__main__":
    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_samples/train")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        for frame in load_frame(scene_path):
            print(frame.timestamp_micros)

            # Obtain range images from the frame
            (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            # Parse range image for lidar 1
            def _range_image_to_pcd(ri_index = 0):
                points, points_cp = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose,
                    ri_index=ri_index)
                return points, points_cp
            
            # Return of the first 2 lidar scans
            pcd_return1 = _range_image_to_pcd()
            # visualize_pointcloud_return(frame, pcd_return1)
            pcd_return2 = _range_image_to_pcd(1)
            # visualize_pointcloud_return(frame, pcd_return2)

            # concatenate 1st and 2nd return
            points, _ = concatenate_pcd_returns(pcd_return1, pcd_return2)
            show_point_cloud(points)
