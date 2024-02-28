"""
This script parses the maps in waymo dataset along with the 3D sem seg labels to generate and enrich the HD maps.
"""

import os
import sys

import numpy as np
import plotly.graph_objs as go
import pathlib
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *

def project_points_on_map(points):
    # Get pointcloud coordinated to project in the map
        xyz = points[0][:, 3:]
        num_points = xyz.shape[0]

        # Transform the points from the vehicle frame to the world frame.
        xyz = np.concatenate([xyz, np.ones([num_points, 1])], axis=-1)
        transform = np.reshape(np.array(frame.pose.transform), [4, 4])
        xyz = np.transpose(np.matmul(transform, np.transpose(xyz)))[:, 0:3]

        # Correct the pose of the points into the coordinate system of the first
        # frame to align with the map data.
        offset = frame.map_pose_offset
        points_offset = np.array([offset.x, offset.y, offset.z])
        xyz += points_offset

        return xyz


if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_samples")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        print("Scene {} processing: {}".format(str(scene_index), scene_path))
        # Initialize frame with map information
        frame_id = 0
        # map_features = 
        for frame in load_frame(scene_path):
            ## For the first frame > Only retreive frame with map information
            ## Save map features in a variable to use it with the semantic information from the LiDAR
            if hasattr(frame, 'map_features') and frame.map_features:
                # Retrieve map_feature in the firts frame
                map_features = frame.map_features

            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            # Only generate information of 3D labeled frames
            if not(segmentation_labels):
                continue
                
            # Only get the first frame of the scene with segmentation labels
            if frame_id == 0:
                frame_id += 1
                # Project the range images into points.
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame,
                    range_images,
                    camera_projections,
                    range_image_top_pose,
                    keep_polar_features=True,
                )

                #  Get the labeled points from the point cloud
                point_labels = convert_range_image_to_point_cloud_labels(
                    frame, range_images, segmentation_labels)
                
                filtered_point_cloud, filtered_point_labels = filter_lidar_data(points, point_labels)

                # Get projection of LiDAR points on the map
                xyz = project_points_on_map(filtered_point_cloud)

                # Plot the point cloud for this frame aligned with the map data.
                figure = plot_maps.plot_map_features(map_features)

                intensity = filtered_point_cloud[0][:, 0]

                # Class Colors
                non_empty_labels = [arr for arr in filtered_point_labels if arr.size != 0]
                # Get only segmentation labels (not instance)
                filtered_point_labels = np.concatenate(non_empty_labels, axis=0)[:,1]
                # unique_segment_ids = np.unique(filtered_point_labels[:, 1])
                # class_color_mapping = {class_id: plt.cm.tab20(i) for i, class_id in enumerate(unique_segment_ids)}

                figure.add_trace(
                    go.Scatter3d(
                        x=xyz[:, 0],
                        y=xyz[:, 1],
                        z=xyz[:, 2],
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=filtered_point_labels,  # set color to an array/list of desired values
                            colorscale='Pinkyl',  # choose a colorscale
                            opacity=0.8,
                        ),
                    )
                )

                figure.show()

