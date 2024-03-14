"""
This script parses the maps in waymo dataset along with the 3D sem seg labels to generate and enrich the HD maps.
"""

import os
import sys

import numpy as np
import plotly.graph_objs as go
import pathlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan

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
    xyz = points[0]
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

# def project_points_on_map(points):
#     # Get pointcloud coordinated to project in the map
#     num_points = points.shape[0]

#     # Transform the points from the vehicle frame to the world frame.
#     points = np.concatenate([points, np.ones([num_points, 1])], axis=-1)
#     transform = np.reshape(np.array(frame.pose.transform), [4, 4])
#     points = np.transpose(np.matmul(transform, np.transpose(points)))[:, 0:3]

#     # Correct the pose of the points into the coordinate system of the first
#     # frame to align with the map data.
#     offset = frame.map_pose_offset
#     points_offset = np.array([offset.x, offset.y, offset.z])
#     points += points_offset

#     return points


def get_color_palette_from_labels(labels):
    # Class Colors
    non_empty_labels = [arr for arr in labels if arr.size != 0]
    # Get only segmentation labels (not instance)
    labels = np.concatenate(non_empty_labels, axis=0)

    return labels


def plot_pointcloud_on_map(map, point_cloud, point_cloud_labels):
    # # Get color palette from labels to plot
    # if (len(point_cloud_labels) > 0):
    #     color_palette = get_color_palette_from_labels(point_cloud_labels)

    color_palette = point_cloud_labels

    # Plot the point cloud for this frame aligned with the map data.
    figure = plot_maps.plot_map_features(map)

    scatter_trace = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=color_palette,
            colorscale='Pinkyl',
            opacity=0.8,
        ),
    )

    figure.add_trace(scatter_trace)

    # Add bounding boxes
    for label in np.unique(point_cloud_labels):
        cluster_points = point_cloud[point_cloud_labels == label]

        # Compute bounding box
        min_bound = np.min(cluster_points, axis=0)
        max_bound = np.max(cluster_points, axis=0)

        # Create bounding box trace
        bbox_trace = go.Mesh3d(
            x=[min_bound[0], max_bound[0], max_bound[0], min_bound[0], min_bound[0],
               min_bound[0], max_bound[0], max_bound[0], min_bound[0], min_bound[0],
               min_bound[0], max_bound[0], max_bound[0], min_bound[0], min_bound[0],
               min_bound[0], max_bound[0], max_bound[0], min_bound[0], min_bound[0],
               min_bound[0], max_bound[0], max_bound[0], min_bound[0]],
            y=[min_bound[1], min_bound[1], max_bound[1], max_bound[1], min_bound[1],
               min_bound[1], min_bound[1], max_bound[1], max_bound[1], max_bound[1],
               min_bound[1], min_bound[1], max_bound[1], max_bound[1], min_bound[1],
               min_bound[1], min_bound[1], min_bound[1], max_bound[1], max_bound[1],
               min_bound[1], min_bound[1], max_bound[1], max_bound[1]],
            z=[min_bound[2], min_bound[2], min_bound[2], min_bound[2], min_bound[2],
               max_bound[2], max_bound[2], max_bound[2], max_bound[2], max_bound[2],
               max_bound[2], max_bound[2], max_bound[2], max_bound[2], max_bound[2],
               min_bound[2], min_bound[2], max_bound[2], max_bound[2], max_bound[2],
               min_bound[2], min_bound[2], min_bound[2], min_bound[2]],
            opacity=0.5,
            color='red',
            name=f'Cluster {label} Bounding Box'
        )

        figure.add_trace(bbox_trace)

    figure.show()

def get_differentiated_colors(numbers):
    # Choose a colormap (you can change this to any other colormap available in matplotlib)
    colormap = cm.get_cmap('viridis')

    # Normalize the numbers to map them to the colormap range
    normalize = Normalize(vmin=min(numbers), vmax=max(numbers))

    # Map each number to a color in the chosen colormap
    colors = [colormap(normalize(num)) for num in numbers]

    return colors


if __name__ == "__main__":
    from sklearn.neighbors import NearestNeighbors
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        print("Scene {} processing: {}".format(str(scene_index), scene_path))

        # Array to store segmented pointclouds
        point_clouds = []
        # Array to store pointcloud labels
        point_cloud_labels = []

        for frame in load_frame(scene_path):
            # Get Map Information of the scene
            # For the first frame > Only retreive frame with map information
            # Save map features in a variable to use it with the semantic information from the LiDAR
            if hasattr(frame, 'map_features') and frame.map_features:
                # Retrieve map_feature in the firts frame
                map_features = frame.map_features
                print("Map feature found in scene")
            
            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            # Only generate information of 3D labeled frames
            if not(segmentation_labels):
                continue
            print("Segmentation label found")

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
            
            filtered_point_cloud, filtered_point_labels = filter_lidar_data(points_return1, point_labels, [8, 10])

            projected_point_cloud = project_points_on_map(filtered_point_cloud)

            # # Concatenate points of the 5 LiDAR
            # concat_point_cloud = concatenate_points(projected_point_cloud)
            # Concatenate labels of points of the 5 LiDAR
            concat_point_labels = concatenate_points(filtered_point_labels)

            # Get semantic and instance segmentation labels
            semantic_labels = concat_point_labels[:,1]
            instance_labels = concat_point_labels[:,0]

            point_clouds.append(projected_point_cloud)
            point_cloud_labels.append(semantic_labels)

        # Concatenate pointclouds of the scene
        if len(point_clouds) > 1:
            point_clouds = np.concatenate(point_clouds, axis=0)
            point_cloud_labels = np.concatenate(point_cloud_labels, axis=0)
        else:
            print("No pointclouds in scene")
            continue

        # # Find best epsilon for DBSCAN
        # neigh = NearestNeighbors(n_neighbors=2)
        # nbrs = neigh.fit(point_clouds[:,:2])
        # distances, indices = nbrs.kneighbors(point_clouds[:,:2])
        # # Plotting K-distance Graph
        # distances = np.sort(distances, axis=0)
        # distances = distances[:,1]
        # plt.figure(figsize=(20,10))
        # plt.plot(distances)
        # plt.title('K-distance Graph',fontsize=20)
        # plt.xlabel('Data Points sorted by distance',fontsize=14)
        # plt.ylabel('Epsilon',fontsize=14)
        # plt.show()

        clustering = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05).fit(point_clouds[:,:2])
        cluster_labels = clustering.labels_[clustering.ordering_]

        print("Clusters detected: ", len(np.unique(cluster_labels)))
        print("Segmentation labels: ", cluster_labels)
        print("Clustered labels: ", point_cloud_labels)

        colors = get_differentiated_colors(cluster_labels)

        # Plot Map and PointCloud aligned with the map data.
        plt.figure()
        plt.scatter(point_clouds[:,0], point_clouds[:,1], color=colors)
        plt.show()
        plot_pointcloud_on_map(map_features, point_clouds, cluster_labels)



