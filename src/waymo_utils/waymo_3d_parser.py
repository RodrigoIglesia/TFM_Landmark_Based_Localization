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
import tensorflow as tf
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan


if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


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


def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
    """Plots range image.

    Args:
        data: range image data
        name: the image title
        layout: plt layout
        vmin: minimum value of the passed data
        vmax: maximum value of the passed data
        cmap: color map
    """
    plt.subplot(*layout)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')


def show_semseg_label_image(semseg_label_image, layout_index_start = 1):
    """Shows range image.

    Args:
        show_semseg_label_image: the semseg label data of type MatrixInt32.
        layout_index_start: layout offset
    """
    semseg_label_image_tensor = tf.convert_to_tensor(semseg_label_image.data)
    semseg_label_image_tensor = tf.reshape(
        semseg_label_image_tensor, semseg_label_image.shape.dims)
    instance_id_image = semseg_label_image_tensor[...,0] 
    semantic_class_image = semseg_label_image_tensor[...,1]
    plot_range_image_helper(instance_id_image.numpy(), 'instance id',
                    [8, 1, layout_index_start], vmin=-1, vmax=200, cmap='Paired')
    plot_range_image_helper(semantic_class_image.numpy(), 'semantic class',
                    [8, 1, layout_index_start + 1], vmin=0, vmax=22, cmap='tab20')


def concatenate_points(points):
    # Concatenate only non-empty arrays
    non_empty_points = [arr for arr in points if arr.size != 0]
    points_all = np.concatenate(non_empty_points, axis=0)
    # points_all = np.concatenate(points, axis=0)
    print(f'points_all shape: {points_all.shape}')

    return points_all


def show_point_cloud_with_labels(points, segmentation_labels):
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    point_cloud.points = o3d.utility.Vector3dVector(points)

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
    Function to filter points from  in the no label 
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
    clustering = DBSCAN(eps=2, min_samples=1).fit(point_cloud[:,:2])
    cluster_labels = clustering.labels_

    print("Clusters detected: ", len(np.unique(cluster_labels)))
    print("Segmentation labels: ", cluster_labels)
    print("Clustered labels: ", cluster_labels)

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

    
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([point_cloud],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])


if __name__ == "__main__":
    from WaymoParser import *
    from sklearn.cluster import DBSCAN


    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        for frame in load_frame(scene_path):
            print(frame.timestamp_micros)

            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
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
            
            filtered_point_cloud, filtered_point_labels = filter_lidar_data(points_return1, point_labels, [8, 10])

            # Concatenate points of the 5 LiDAR
            concat_point_cloud = concatenate_points(filtered_point_cloud)
            # Concatenate labels of points of the 5 LiDA
            concat_point_labels = concatenate_points(filtered_point_labels)

            # Get semantic and instance segmentation labels
            concat_semantic_labels = concat_point_labels[:,1]
            concat_instance_labels = concat_point_labels[:,0]

            # # Cluster filtered pointcloud
            clustered_point_clouds, cluster_labels = cluster_pointcloud(concat_point_cloud)

            # plt.figure()
            # plt.scatter(concat_point_cloud[:,0], concat_point_cloud[:,1], c=cluster_labels)
            # plt.show()

            show_point_cloud_with_labels(concat_point_cloud, cluster_labels)
