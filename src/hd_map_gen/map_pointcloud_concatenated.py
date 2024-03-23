"""
Author: Rodrigo de la Iglesia Sánchez.
Date: 23/03/2024.

This script parses the maps in Waymo dataset along with the 3D sem seg labels to generate and enrich the HD maps.
For each scene:
1. Retrieve the Feature Map, which is contained in the first frame of the scene.
    1.1. If no pointclous is found in the scene, jump to the next scene
2. Point cloud read and clustering
    2.1. Read the 3D pointcloud for each frame.
    2.2. If the frame has pointcloud with segmentation labels > Get a pointcloud only containing points labeled as signs.
    2.3. Save each frame pointcloud in a vector.
    2.4. Concatenate the pointcloud vectors.
    2.5. Apply a clustering algorithm to the pointcloud vector,
    obtaining a the point cloud vector divided in classes (clustered point cloud) and a vector with a class for each point in the point cloud.
3. Enrich the feature map.
    3.1. For each cluster in the clustered pointcloud, compute the centroid of the cluster and project the point to the ground.
    3.2. Generate a protocol buffer message (type StopSign) with the coordinates of the centroid and the lane value to 100 (vertical sign) and add it to the feature map.
    3.3. Save the new feature map as a JSON.
4. Generate a new tfrecord file.
    4.1. Parse the scene dataset's frames.
    4.2. For the first frame > change the feature map for the new one with signs.
    4.3. Serialize each frame to a String an write it in the new tfrecord file.
    4.4. Finally, save the generated tfrecord file.
"""

import os
import sys

import logging
import numpy as np
import plotly.graph_objs as go
import pathlib

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

# Configure logging
logging.basicConfig(filename='logs/waymo_hd_map_gen.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')


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

    # Append the new map feature object in the existing map feature
    map_features.append(new_map_feature)


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
        # Extract x, y, z coordinates from vertices
        x_coords, y_coords, z_coords = zip(*vertices)

        # Create trace for cube edges
        edge_top = go.Scatter3d(
            x=[x_coords[0], x_coords[1], x_coords[2], x_coords[3], x_coords[0]],
            y=[y_coords[0], y_coords[1], y_coords[2], y_coords[3], y_coords[0]],
            z=[z_coords[0], z_coords[1], z_coords[2], z_coords[3], z_coords[0]],
            mode='lines',
            line=dict(color='yellow', width=2)
        )
        figure.add_trace(edge_top)

        edge_bottom = go.Scatter3d(
            x=[x_coords[4], x_coords[5], x_coords[6], x_coords[7], x_coords[4]],
            y=[y_coords[4], y_coords[5], y_coords[6], y_coords[7], y_coords[4]],
            z=[z_coords[4], z_coords[5], z_coords[6], z_coords[7], z_coords[4]],
            mode='lines',
            line=dict(color='yellow', width=2)
        )
        figure.add_trace(edge_bottom)

        edge_sides = go.Scatter3d(
            x=[x_coords[0], x_coords[4], x_coords[5], x_coords[1], x_coords[2], x_coords[6], x_coords[7], x_coords[3]],
            y=[y_coords[0], y_coords[4], y_coords[5], y_coords[1], y_coords[2], y_coords[6], y_coords[7], y_coords[3]],
            z=[z_coords[0], z_coords[4], z_coords[5], z_coords[1], z_coords[2], z_coords[6], z_coords[7], z_coords[3]],
            mode='lines',
            line=dict(color='yellow', width=2)
        )
        figure.add_trace(edge_sides)

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


def save_protobuf_features(protobuf_message, output):
    json_data = []
    for map_feature in protobuf_message:
        json_data.append(json.loads(MessageToJson(map_feature)))  # Convert JSON string to dictionary

    with open(output, 'w') as json_file:
        json.dump(json_data, json_file, separators=(',', ':'))

    logging.info("Protobuf data converted to JSON and saved.")


if __name__ == "__main__":
    ##############################################################
    ## Load Dataset to generate landmarks of the signs
    ##############################################################
    dataset_path        = os.path.join(src_dir, "dataset/waymo_samples")
    json_maps_path      = os.path.join(src_dir, "dataset/hd_maps")
    point_clouds_path   = os.path.join(src_dir, "dataset/pointclouds")
    output_dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene_mod")

    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    ## Iterate dataset
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        logging.info("Scene {} processing: {}".format(str(scene_index), scene_path))

        # Array to store segmented pointclouds
        point_clouds = []
        # Array to store pointcloud labels
        point_cloud_labels = []

        map_features_found = False

        for frame in load_frame(scene_path):
            # Get Map Information of the scene
            # For the first frame > Only retreive frame with map information
            # Save map features in a variable to use it with the semantic information from the LiDAR
            if hasattr(frame, 'map_features') and frame.map_features:
                # Retrieve map_feature in the firts frame
                map_features = frame.map_features
                map_features_found = True
                logging.info("Map feature found in scene, processing point clouds...")
        # If no map features in the scene, jump to the next one
        if (map_features_found == False):
            logging.info("No Map Features found in the scene, jumping to the next one...")
            continue
        
        # If map features were found, parse the 3D point clouds in the frames
        for frame in load_frame(scene_path):
            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            # Only generate information of 3D labeled frames
            if not(segmentation_labels):
                continue
            logging.debug("Segmentation label found")

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
            logging.debug("Scene pointclouds correctly concatenated")
        else:
            logging.debug("No pointclouds in scene")
            continue

        # Save point cloud to csv
        out_csv_path = point_clouds_path + '/pointcloud_concatenated' + os.path.splitext(os.path.basename(scene_path))[0] + '.csv'
        # Use savetxt to save the array to a CSV file
        np.savetxt(out_csv_path, point_clouds, delimiter=',')
        logging.debug(f"NumPy array has been successfully saved to {out_csv_path}.")

        # Get the clustered pointclouds, each cluster corresponding to a traffic sign
        clustered_point_cloud, cluster_labels = cluster_pointcloud(point_clouds)

        # Add signs to map
        sign_id = map_features[-1].id
        for cluster in clustered_point_cloud:
            # Get the centroid of each cluster of the pointcloud
            cluster_centroid = get_cluster_centroid(cluster)

            # Add sign centroids to feature map
            add_sign_to_map(map_features, cluster_centroid, sign_id)
            logging.debug("Sign message added to map features")
            sign_id += 1
        save_protobuf_features(map_features, json_maps_path + "/signs_map_features_" + os.path.splitext(os.path.basename(scene_path))[0] + '.json')
        logging.debug("Modified map saved as JSON")

        ## Save scene as tfrecord
        output_filename = output_dataset_path + '/output' + os.path.basename(scene_path)
        writer = tf.io.TFRecordWriter(output_filename)
        for frame in load_frame(scene_path):
            if hasattr(frame, 'map_features') and frame.map_features:
                logging.debug("Removing current map features")
                # Retrieve map_feature in the firts 
                del frame.map_features[:]

                # Append the new map_features object to the cleared list
                logging.debug("Adding modified map features")
                for feature in map_features:
                    frame.map_features.append(feature)
            serialized_frame = frame.SerializeToString()
            writer.write(serialized_frame)
            logging.info("Tfrecord saved...")

        # Close the writer
        writer.close()

        # plot_pointcloud_on_map(map_features, point_clouds, cluster_labels)