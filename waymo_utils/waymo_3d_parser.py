"""
Waymo Open Dataset - 3D Semantic Segmentation parser (LiDAR)
Este módulo contiene funciones para:
Leer cada frame del dataset y extraer la nube de puntos
En algunos frames hay segmentación semántica de las nubes de puntos
En caso de existir, se muestra en 3D
"""

import os
import sys

import matplotlib.pyplot as plt
import pathlib
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


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


def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """
    Convert segmentation labels from range images to point clouds.
    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
            
        point_labels.append(sl_points_tensor.numpy())

    return point_labels


def concatenate_pcd_returns(pcd_return_1, pcd_return_2):
    points, points_cp = pcd_return_1
    points_ri2, points_cp_ri2 = pcd_return_2
    points_concat = np.concatenate(points + points_ri2, axis=0)
    points_cp_concat = np.concatenate(points_cp + points_cp_ri2, axis=0)

    return points_concat, points_cp_concat


def show_semseg_label_image(semseg_label_image, layout_index_start = 1):
    """Shows range image.

    Args:
        show_semseg_label_image: the semseg label data of type MatrixInt32.
        layout_index_start: layout offset
    """
    semseg_label_image_tensor = tf.convert_to_tensor(semseg_label_image.data)
    semseg_label_image_tensor = tf.reshape(
        semseg_label_image_tensor, semseg_label_image.shape.dims)
    range_image_range = semseg_label_image_tensor[...,0] 
    range_image_intensity = semseg_label_image_tensor[...,1]
    range_image_elongation = semseg_label_image_tensor[...,2]

    plot_range_image_helper(range_image_range.numpy(), 'Range',
                    [8, 1, layout_index_start], vmax=75, cmap='Paired')
    plot_range_image_helper(range_image_intensity.numpy(), 'Intensity',
                    [8, 1, layout_index_start + 1], vmax=1.5, cmap='tab20')
    plot_range_image_helper(range_image_elongation.numpy(), 'Elongation',
                    [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')


def concatenate_points(points):
    # Concatenate only non-empty arrays
    non_empty_points = [arr for arr in points if arr.size != 0]
    points_all = np.concatenate(non_empty_points, axis=0)
    # points_all = np.concatenate(points, axis=0)

    return points_all


def show_point_cloud_with_labels(points, segmentation_labels=None):
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Move the origin closer to the point cloud coordinates
    origin = np.mean(points, axis=0)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=origin)

    point_cloud.points = o3d.utility.Vector3dVector(points)

    if segmentation_labels:
        # Labels
        # Extract unique class IDs and instance IDs
        unique_segment_ids = np.unique(segmentation_labels)

        # Create a color mapping based on class IDs
        class_color_mapping = {class_id: plt.cm.tab20(i) for i, class_id in enumerate(unique_segment_ids)}

        # Create a color array based on segmentation labels
        colors = np.array([class_color_mapping[class_id] for class_id in segmentation_labels])


        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

        vis.add_geometry(point_cloud)
        # Plot bounding boxes for each cluster
        for label in np.unique(segmentation_labels):
            cluster_points = points[segmentation_labels == label]

            # Compute bounding box
            min_bound = np.min(cluster_points, axis=0)
            max_bound = np.max(cluster_points, axis=0)

            # Create bounding box
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            bbox_wireframe = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
            bbox_wireframe.paint_uniform_color([1, 0, 0])  # Red color

            # Add bounding box to the visualizer
            vis.add_geometry(bbox_wireframe)
    vis.add_geometry(mesh_frame)

    vis.run()


def filter_lidar_data(point_clouds, segmentation_labels, labels_to_keep):
    """
    Function to filter points.
    """
    combined_data = list(zip(point_clouds, segmentation_labels))

    # Filter out points with any segmentation label being 0
    filtered_points = []
    filtered_labels = []
    for lidar_data in combined_data:
        filtered_lidar_point = []
        filtered_lidar_label = []
        for point, label in zip(lidar_data[0], lidar_data[1]):
            if (not np.any(label == 0) and (label[1] in labels_to_keep)):
            # if (not np.any(label == 0)):
                filtered_lidar_point.append(point)
                filtered_lidar_label.append(label)
            else: continue
        filtered_points.append(np.array(filtered_lidar_point))
        filtered_labels.append(np.array(filtered_lidar_label))

    return filtered_points, filtered_labels


def cluster_pointcloud(point_cloud):
    # clustering = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05).fit(point_clouds[:,:2])
    clustering = DBSCAN(eps=0.15, min_samples=1).fit(point_cloud[:,:2])
    cluster_labels = clustering.labels_

    clustered_point_cloud = []
    for label in np.unique(cluster_labels):
        cluster_points = point_cloud[cluster_labels == label]
        clustered_point_cloud.append(cluster_points)

    return clustered_point_cloud, cluster_labels


def calculate_centroid(points):
    """
    Calculate centroid of points.
    Args:
    - points: numpy array of shape (N, 3) representing points
    Returns:
    - centroid: numpy array of shape (3,) representing centroid
    """

    centroid = np.mean(points, axis=0)
    return centroid


def get_cluster_centroid(point_cloud):
    # Define ground normal (example: [0, 1, 0] for a horizontal ground)
    min_z = np.min(point_cloud[:,2], axis=0)

    # Calculate centroid of the projected points
    centroid = calculate_centroid(point_cloud)
    centroid_projected = np.array([centroid[0], centroid[1], min_z])

    return centroid_projected


def get_cluster_orientation(point_cloud):
    """
    Get roll, pitch and yaw of a pointcloud
    """
    points_np = np.array(point_cloud)
    # Calculate orientation using PCA
    cov_matrix = np.cov(points_np.T)  # Compute covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # Eigen decomposition
    rotation_matrix = eigvecs[:, ::-1]  # Align with major axis (descending order of eigenvalues)

    # Convert rotation matrix to roll, pitch, yaw
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)  # Convert to Euler angles

    return [roll,pitch,yaw]


def plot_referenced_pointcloud(point_cloud, size=0.6, color=None, plot=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if color is not None:
        pcd.paint_uniform_color(color)  # Set the color of the points
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])

    if (plot):
        o3d.visualization.draw_geometries([pcd, mesh_frame])
    else:
        return [pcd, mesh_frame]


def plot_labeled_pointcloud(labeled_pointclouds):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for color, points in labeled_pointclouds.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        color_float = tuple(np.array(color) / 255.0)
        pcd.paint_uniform_color(color_float)

        vis.add_geometry(pcd)

        # Compute bounding box
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        # Create bounding box
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_wireframe = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
        bbox_wireframe.paint_uniform_color([1, 0, 0])  # Red color

        # Add bounding box to the visualizer
        vis.add_geometry(bbox_wireframe)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    from WaymoParser import *
    from sklearn.cluster import DBSCAN


    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
    sys.path.append(src_dir)

    scene_path = os.path.join(src_dir, "dataset/final_tests_scene/individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord")

    for frame_index, frame in enumerate(load_frame(scene_path)):
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        if not(segmentation_labels):
            continue

        frame.lasers.sort(key=lambda laser: laser.name)
        show_semseg_label_image(range_images[open_dataset.LaserName.TOP][0])
        plt.show()


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
        # plot_referenced_pointcloud(filtered_point_cloud[0], plot=True)

        # Concatenate points of the 5 LiDAR
        concat_point_cloud = concatenate_points(filtered_point_cloud)
        # Concatenate labels of points of the 5 LiDA
        concat_point_labels = concatenate_points(filtered_point_labels)

        # Get semantic and instance segmentation labels
        concat_semantic_labels = concat_point_labels[:,1]
        concat_instance_labels = concat_point_labels[:,0]

        # # Cluster filtered pointcloud
        clustered_point_clouds, cluster_labels = cluster_pointcloud(concat_point_cloud)

        plt.figure()
        plt.scatter(concat_point_cloud[:,0], concat_point_cloud[:,1], c=cluster_labels)
        plt.title(f"Clustering Visualization for Frame {frame_index}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

        # show_point_cloud_with_labels(concat_point_cloud, concat_semantic_labels)
