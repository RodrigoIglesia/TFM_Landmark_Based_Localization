"""
Author: Rodrigo de la Iglesia Sánchez.
Date: 23/03/2024.

This script parses the maps in Waymo dataset along with the 3D sem seg labels to generate and enrich the HD maps.
For each scene:
1. Retrieve the Feature Map, which is contained in the first frame of the scene.
    1.1. If no pointclous is found in the scene, jump to the next scene
2. Get Scene Labeled Point
    2.1. Read the 3D pointcloud for each frame.
    2.2. If the frame has pointcloud with segmentation labels > Get a pointcloud only containing points labeled as signs.
    2.3. Save each frame pointcloud in a vector.
    2.4. Concatenate the pointcloud vectors > to do that, all pointclouds must be expressed in the same reference frame
        For this implementation, the global frame used is the origin of movement of the vehicle in the scene (instead of waymo global frame)
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
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pathlib

import json
import csv
from google.protobuf.json_format import MessageToJson

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)
print(src_dir)

import waymo_utils.transform_utils as tu

###############################################################################
## Configure logging
###############################################################################

# Set up logging to both file and console
log_file = src_dir + '/logs/waymo_hd_map_gen.log'

# Create handlers
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler(sys.stdout)

# Set logging level for both handlers
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *


def create_homog_matrix(position):
    """
    Creates a homogeneous transformation matrix (4x4) from a position given as x, y, z, roll, pitch, yaw.
    :param position: Dictionary with keys ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    :return: 4x4 numpy array representing the homogeneous transformation matrix.
    """
    x = position['x']
    y = position['y']
    z = position['z']
    roll = position['roll']
    pitch = position['pitch']
    yaw = position['yaw']

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Homogeneous transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[0, 3] = x
    transform[1, 3] = y
    transform[2, 3] = z

    return transform

def compose_positions(accumulated_pose, increment):
    """
    Composes two poses given in positionRPY format (x, y, z, roll, pitch, yaw).
    :param accumulated_pose: Dictionary with keys ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    :param increment: Dictionary with keys ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    :return: Dictionary representing the composed pose.
    """
    # Create homogeneous transformation matrices
    actual_matrix = create_homog_matrix(accumulated_pose)
    increment_matrix = create_homog_matrix(increment)

    # Multiply the matrices to compose the pose
    result_matrix = actual_matrix @ increment_matrix

    # Extract translation
    x, y, z = result_matrix[:3, 3]

    # Extract rotation
    R = result_matrix[:3, :3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return {
        'x': x,
        'y': y,
        'z': z,
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw
    }

def project_points_on_map(pointcloud, transformation_matrix):
    """
    Transforms a point cloud using a homogeneous transformation matrix.
    :param pointcloud: Nx3 numpy array representing the point cloud.
    :param transformation_matrix: 4x4 numpy array representing the transformation matrix.
    :return: Transformed Nx3 numpy array.
    """
    # Convert the point cloud to homogeneous coordinates
    ones = np.ones((pointcloud.shape[0], 1))
    homogeneous_points = np.hstack((pointcloud, ones))

    # Apply the transformation
    transformed_points = (transformation_matrix @ homogeneous_points.T).T

    # Return the points in 3D coordinates
    return transformed_points[:, :3]


def matrix_to_pose(transformation_matrix):
    """
    Converts a 4x4 homogeneous transformation matrix into a pose dictionary with keys ['x', 'y', 'z', 'roll', 'pitch', 'yaw'].
    :param transformation_matrix: 4x4 numpy array representing the transformation matrix.
    :return: Dictionary representing the pose.
    """
    # Extract translation
    x, y, z = transformation_matrix[:3, 3]

    # Extract rotation
    R = transformation_matrix[:3, :3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return {
        'x': x,
        'y': y,
        'z': z,
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw
    }


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
    scene_path = os.path.join(src_dir, "dataset/final_tests_scene/individual_files_validation_segment-10247954040621004675_2180_000_2200_000_with_camera_labels.tfrecord")
    json_maps_path      = os.path.join(src_dir, "dataset/hd_maps")
    point_clouds_path   = os.path.join(src_dir, "dataset/pointclouds")
    output_dataset_path = os.path.join(src_dir, "dataset/final_output_scenes")
    output_csv_path     = os.path.join(src_dir, "pointcloud_clustering/map")


    ##############################################################
    ## Iterate through Dataset Scenes
    ##############################################################
    scene_name = pathlib.Path(scene_path).stem
    logging.info("Scene processing: {}".format(scene_path))

    ##############################################################
    ## Retrieve the Feature Map
    ##############################################################
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
    
    ##############################################################
    ## Get Scene Labeled Point
    ##############################################################
    # If map features were found, parse the 3D point clouds in the frames
    # Array to store segmented pointclouds
    point_clouds = []
    # Array to store pointcloud labels
    point_cloud_labels = []

    composed_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
    previous_transform_matrix = None
    for frame_index, frame in enumerate(load_frame(scene_path)):
        ############################################################
        ## Accumulate Pose
        ###########################################################
        current_transform_matrix = np.array(frame.pose.transform).reshape(4, 4)
        if previous_transform_matrix is not None:
            relative_transform = np.linalg.inv(previous_transform_matrix) @ current_transform_matrix
            increment = matrix_to_pose(relative_transform)
            composed_pose = compose_positions(composed_pose, increment)

        previous_transform_matrix = current_transform_matrix
        ############################################################
        ## Process Pointclouds
        ###########################################################
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

        # plot_pointcloud_on_map(map_features, points_return1, point_labels)

        filtered_point_cloud, filtered_point_labels = filter_lidar_data(points_return1, point_labels, [8,9,10])

        ############################################################
        ## Get transform from current pose to vehicle origin pose
        ###########################################################
        transformation_matrix = create_homog_matrix(composed_pose)
        projected_pointcloud = project_points_on_map(filtered_point_cloud[0], transformation_matrix)
        point_clouds.append(projected_pointcloud)

        # Concatenate labels of points of the 5 LiDAR
        concat_point_labels = concatenate_points(filtered_point_labels)

        # Get semantic and instance segmentation labels
        semantic_labels = concat_point_labels[:,1]
        instance_labels = concat_point_labels[:,0]
        point_cloud_labels.append(semantic_labels)

    # Concatenate pointclouds of the scene
    if len(point_clouds) > 1:
        point_clouds = np.concatenate(point_clouds, axis=0)
        point_cloud_labels = np.concatenate(point_cloud_labels, axis=0)
        logging.debug("Scene pointclouds correctly concatenated")
    else:
        logging.debug("No pointclouds in scene")

    # Save point cloud to csv
    out_csv_path = point_clouds_path + '/pointcloud_concatenated' + os.path.splitext(os.path.basename(scene_path))[0] + '.csv'
    # Use savetxt to save the array to a CSV file
    np.savetxt(out_csv_path, point_clouds, delimiter=',')
    logging.debug(f"NumPy array has been successfully saved to {out_csv_path}.")

    # Get the clustered pointclouds, each cluster corresponding to a traffic sign
    clustered_point_cloud, cluster_labels = cluster_pointcloud(point_clouds)


    ##############################################################
    ## Enrich Feature Map
    ##############################################################
    # Add signs to map
    sign_id = map_features[-1].id
    for cluster in clustered_point_cloud:
        # Get the centroid of each cluster of the pointcloud
        cluster_centroid = get_cluster_centroid(cluster)

        # Add sign centroids to feature map
        add_sign_to_map(map_features, cluster_centroid, sign_id)
        logging.debug("Sign message added to map features")
        sign_id += 1


    json_file = json_maps_path + "/signs_map_features_" + os.path.splitext(os.path.basename(scene_path))[0] + '.json'
    save_protobuf_features(map_features, json_file)
    logging.debug("Modified map saved as JSON")


    ##############################################################
    ## Save map in csv format
    ##############################################################
    with open(json_file, 'r') as f:
        data = json.load(f)
    output_csv = output_csv_path + "/signs_map_features_" + os.path.splitext(os.path.basename(scene_path))[0] + '.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Iterate through each item in the JSON
        for item in data:
            if 'stopSign' in item:
                stop_sign = item['stopSign']
                if 'lane' in stop_sign and stop_sign['lane'] == ["100"]:
                    # Extract x and y coordinates from position
                    position = stop_sign['position']
                    csv_writer.writerow([position['x'], position['y'], position['z']])



    # ##############################################################
    # ## Generate a new tfrecord file
    # ##############################################################
    # output_filename = output_dataset_path + '/output' + os.path.basename(scene_path)
    # writer = tf.io.TFRecordWriter(output_filename)
    # for frame in load_frame(scene_path):
    #     if hasattr(frame, 'map_features') and frame.map_features:
    #         logging.debug("Removing current map features")
    #         # Retrieve map_feature in the firts 
    #         del frame.map_features[:]

    #         # Append the new map_features object to the cleared list
    #         logging.debug("Adding modified map features")
    #         for feature in map_features:
    #             frame.map_features.append(feature)
    #     serialized_frame = frame.SerializeToString()
    #     writer.write(serialized_frame)
    #     logging.info("Tfrecord saved...")

    # # Close the writer
    # writer.close()

    plot_pointcloud_on_map(map_features, point_clouds, cluster_labels)