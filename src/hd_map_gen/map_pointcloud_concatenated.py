"""
This script parses the maps in waymo dataset along with the 3D sem seg labels to generate and enrich the HD maps.
"""

#TODO: Comprobar que el landmark se genera bien como stop sign
#TODO: Revisar el algoritmo de clustering de la nube de puntos
#TODO: Si lo anterior funciona bien > investigar cómo meter el feature_map modificado en el frame, luego en la escena y generar el tfrecord

import os
import sys

import numpy as np
import plotly.graph_objs as go
import pathlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import json
from google.protobuf.json_format import MessageToJson

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *



def project_points_on_map(points, frame):
    """
    Project coordinates of the point cloud (referenced to the sensor system) to the map (referenced to the world system)
    Args
    - points: 
    """
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


def add_sign_to_map(map_features, sign_coords, id):
    # Create a new map features object to insert the sign there
    new_map_feature = map_pb2.MapFeature()

    new_map_feature.id = id

    # Create a new Driveway message and populate its polygons
    sign_message = new_map_feature.stop_sign
    # sign_message.lane = 100 # 100 indicates is a vertical sign
    sign_message.lane.append(100)
    sign_message.position.x = sign_coords[0]
    sign_message.position.y = sign_coords[1]
    sign_message.position.z = sign_coords[2]

    # # Create a new Driveway message and populate its polygons
    # sign_message = new_map_feature.stop_sign
    # # sign_message.lane = 100 # 100 indicates is a vertical sign
    # sign_message.position.x = sign_coords[0]
    # sign_message.position.y = sign_coords[1]
    # sign_message.position.z = sign_coords[2]

    # Append the new map feature object in the existing map feature
    map_features.append(new_map_feature)

    return map_features


def plot_poincloud(figure, point_cloud, labels):
    color_palette = labels

    scatter_trace = go.Scatter3d(
        x = point_cloud[:, 0],
        y = point_cloud[:, 1],
        z = point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=color_palette,
            colorscale='hsv',
            opacity=0.8,
        ),
    )

    figure.add_trace(scatter_trace)


def plot_cluster_bbox(figure, point_cloud, labels):
    """
    Args:
        point_cloud: (N, 3) 3D point cloud matrix
    """
    # Add bounding boxes to figure
    for label in np.unique(labels):
        cluster_points = point_cloud[labels == label]

        # Compute bounding box
        min_bound = np.min(cluster_points, axis=0)
        max_bound = np.max(cluster_points, axis=0)

        # Define vertices for all faces of the bounding box
        vertices = [
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]]
        ]

        # Define the indices of vertices for each face of the bounding box
        # Define cube edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # Extract x, y, z coordinates from vertices
        x_coords, y_coords, z_coords = zip(*vertices)

        # Create trace for cube edges
        edge_trace = go.Scatter3d(
            x=[x_coords[i] for i, j in edges] + [None],
            y=[y_coords[i] for i, j in edges] + [None],
            z=[z_coords[i] for i, j in edges] + [None],
            mode='lines',
            line=dict(color='red', width=2)
        )
        figure.add_trace(edge_trace)

        # Create trace for cube vertices
        vertex_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(color='red', size=2)
        )

        figure.add_trace(vertex_trace)

def plot_pointcloud_on_map(map, point_cloud, labels):
    # Plot the point cloud for this frame aligned with the map data.
    figure = plot_maps.plot_map_features(map)

    plot_poincloud(figure, point_cloud, labels)
    plot_cluster_bbox(figure, point_cloud, labels)

    figure.show()


def get_differentiated_colors(numbers):
    # Choose a colormap (you can change this to any other colormap available in matplotlib)
    colormap = cm.get_cmap('viridis')

    # Normalize the numbers to map them to the colormap range
    normalize = Normalize(vmin=min(numbers), vmax=max(numbers))

    # Map each number to a color in the chosen colormap
    colors = [colormap(normalize(num)) for num in numbers]

    return colors


def save_protobuf_features(protobuf_message, output):
    json_data = []
    for map_feature in protobuf_message:
        json_data.append(json.loads(MessageToJson(map_feature)))  # Convert JSON string to dictionary

    with open(output, 'w') as json_file:
        json.dump(json_data, json_file, separators=(',', ':'))

    print("Protobuf data converted to JSON and saved to 'output.json'.")


if __name__ == "__main__":
    ##############################################################
    ## Load Dataset to generate landmarks of the signs
    ##############################################################
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")
    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    ## Iterate dataset
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
                save_protobuf_features(map_features, "dataset/map_features_" + str(scene_index) + ".json")
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

            projected_point_cloud = project_points_on_map(filtered_point_cloud, frame)

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

        # # Save point cloud to csv
        # file_path = 'dataset/pointcloud_concatenated.csv'
        # # Use savetxt to save the array to a CSV file
        # np.savetxt(file_path, point_clouds, delimiter=',')
        # print(f"NumPy array has been successfully saved to {file_path}.")

        # Get the clustered pointclouds, each cluster corresponding to a traffic sign
        clustered_point_cloud, cluster_labels = cluster_pointcloud(point_clouds)

        # Add signs to map
        sign_id = map_features[-1].id
        for cluster in clustered_point_cloud:
            # Get the centroid of each cluster of the pointcloud
            cluster_centroid = get_cluster_centroid(cluster)

            # Add sign centroids to feature map
            signs_map_feature = add_sign_to_map(map_features, cluster_centroid, sign_id)
            sign_id += 1
        save_protobuf_features(signs_map_feature, "dataset/map_features_mod_" + str(scene_index) + ".json")

        # colors = get_differentiated_colors(cluster_labels)

        # Plot Map and PointCloud aligned with the map data.
        # plt.figure()
        # plt.scatter(point_clouds[:,0], point_clouds[:,1], color=colors)
        # plt.show()
        plot_pointcloud_on_map(signs_map_feature, point_clouds, cluster_labels)



