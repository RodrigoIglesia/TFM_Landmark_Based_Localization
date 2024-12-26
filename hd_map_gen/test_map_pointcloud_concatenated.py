import os
import sys

import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pathlib
import pandas as pd
import open3d.visualization.gui as gui
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


def visualize_pointclouds(pointclouds_dict):
    geometries = {}

    # Generar colores aleatorios
    color_map = {}
    for name in pointclouds_dict.keys():
        color = [random.random(), random.random(), random.random()]
        color_map[name] = color

    # Crear geometrías de nubes de puntos
    for name, points in pointclouds_dict.items():
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.paint_uniform_color(color_map[name])  # Asignar color aleatorio
        geometries[name] = pc

    # Crear el origen de coordenadas
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries["Origin"] = origin

    # Crear ventana de visualización personalizada
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Nubes de puntos", 1024, 768)

    for name, geometry in geometries.items():
        vis.add_geometry(name, geometry)

    # Añadir etiquetas 3D simuladas
    for i, (name, color) in enumerate(color_map.items()):
        label_position = np.array([-0.6, 0.6 - 0.2 * i, 0], dtype=np.float32)
        text_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        text_sphere.translate(label_position)
        text_sphere.paint_uniform_color(color)
        vis.add_geometry(f"{name}_sphere", text_sphere)

        # Usar etiquetas de texto
        vis.add_3d_label(label_position, name)  # Simula etiquetas cerca de las esferas

    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


if __name__ == "__main__":
    scene_path = os.path.join(src_dir, "dataset/final_tests_scene/individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord")
    signs_map_file = os.path.join(src_dir, "dataset/pointclouds/pointcloud_concatenatedindividual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.csv")
    signs_map = pd.read_csv(signs_map_file)
    map_points = signs_map.iloc[:, :3].values  # Se asume que las primeras tres columnas son X, Y, Z
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
        if hasattr(frame, 'map_features') and frame.map_features:
            map_features = frame.map_features
            map_features_found = True
            logging.info("Map feature found in scene, processing point clouds...")

    if not map_features_found:
        logging.info("No Map Features found in the scene, jumping to the next one...")

    ##############################################################
    ## Get Scene Labeled Point
    ##############################################################
    point_clouds = []
    point_cloud_labels = []

    composed_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
    previous_transform_matrix = None
    translation_vector = np.array([0.01, 0, 0]) # Para desplazar ligeramente las nubes de puntos que solapan y visualizarlo mejor
    for frame_index, frame in enumerate(load_frame(scene_path)):
        if (frame_index == 1):
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        ############################################################
        ## Process Pointclouds
        ###########################################################
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        if not segmentation_labels:
            continue
        logging.debug("Segmentation label found")

        def _range_image_to_pcd(ri_index=0):
            points, points_cp = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose,
                ri_index=ri_index)
            return points, points_cp

        points_return1, _ = _range_image_to_pcd()
        points_return2, _ = _range_image_to_pcd(1)

        point_labels = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels)
        point_labels_ri2 = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels, ri_index=1)
        
        non_empty_points = [arr for arr in points_return1 if arr.size != 0]
        points_all = np.concatenate(non_empty_points, axis=0)

        filtered_point_cloud, filtered_point_labels = filter_lidar_data(points_return1, point_labels, [8, 9, 10])
        non_empty_points2 = [arr for arr in filtered_point_cloud if arr.size != 0]
        points_all2 = np.concatenate(non_empty_points2, axis=0)
        points_all2 += translation_vector

        current_transform_matrix = np.array(frame.pose.transform).reshape(4, 4)
        if previous_transform_matrix is not None:
            relative_transform = np.linalg.inv(previous_transform_matrix) @ current_transform_matrix
            increment = matrix_to_pose(relative_transform)
            composed_pose = compose_positions(composed_pose, increment)

        previous_transform_matrix = current_transform_matrix
        transformation_matrix = create_homog_matrix(composed_pose)
        projected_pointcloud = project_points_on_map(filtered_point_cloud[0], transformation_matrix)
        projected_pointcloud += translation_vector

        # Visualizar las nubes de puntos
        visualize_pointclouds({"map_points": map_points, "pointcloud_BL": points_all, "filtered_pointcloud_BL": points_all2, "projected_pointcloud_MF": projected_pointcloud})



        point_clouds.append(projected_pointcloud)

        concat_point_labels = concatenate_points(filtered_point_labels)
        semantic_labels = concat_point_labels[:, 1]
        instance_labels = concat_point_labels[:, 0]
        point_cloud_labels.append(semantic_labels)

