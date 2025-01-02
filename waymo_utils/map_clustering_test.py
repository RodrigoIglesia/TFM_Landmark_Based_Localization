import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)
print(src_dir)

import waymo_utils.transform_utils as tu
from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *

if __name__ == "__main__":
    scene_path = os.path.join(src_dir, "dataset/final_tests_scene/individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord")
    ##############################################################
    ## Iterate through Dataset Scenes
    ##############################################################
    scene_name = pathlib.Path(scene_path).stem

    ##############################################################
    ## Get Scene Labeled Point
    ##############################################################
    for frame_index, frame in enumerate(load_frame(scene_path)):
        ############################################################
        ## Process Pointclouds
        ###########################################################
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        # Only generate information of 3D labeled frames
        if not(segmentation_labels):
            continue

        # Get points labeled for first and second return
        # Parse range image for lidar 1
        def _range_image_to_pcd(ri_index = 0):
            points, points_cp = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose,
                ri_index=ri_index)
            return points, points_cp
        
        # Return of the first 2 lidar scans
        points_return1, _ = _range_image_to_pcd()
        points_return2, _ = _range_image_to_pcd(1)

        # Semantic labels for the first 2 lidar scans
        point_labels = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels)
        point_labels_ri2 = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels, ri_index=1)

        filtered_point_cloud, filtered_point_labels = filter_lidar_data(points_return1, point_labels, [8,9,10])
        filtered_point_cloud = filtered_point_cloud[0]
        

        # Get the clustered pointclouds, each cluster corresponding to a traffic sign
        clustering = DBSCAN(eps=0.15, min_samples=1).fit(filtered_point_cloud[:,:2])
        cluster_labels = clustering.labels_
        print(len(cluster_labels))

        clustered_point_cloud = []
        for label in np.unique(cluster_labels):
            cluster_points = filtered_point_cloud[cluster_labels == label]
            clustered_point_cloud.append(cluster_points)

        ############################################################
        ## Visualization of Clustering
        ############################################################
        plt.figure(figsize=(10, 8))

        # Assign a unique color to each cluster
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.get_cmap("tab10", len(unique_labels))

        for label in unique_labels:
            cluster_points = filtered_point_cloud[cluster_labels == label]
            color = colors(label) if label != -1 else "black"  # Black for noise points
            plt.scatter(cluster_points[:, 1], cluster_points[:, 2], c=[color], label=f"Cluster {label}" if label != -1 else "Noise", s=10)

        plt.title(f"Clustering Visualization for Frame {frame_index}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()
