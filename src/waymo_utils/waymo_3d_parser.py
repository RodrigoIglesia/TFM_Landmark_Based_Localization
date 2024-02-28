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


if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

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
    
def visualize_pointcloud_return(points, segmentation_labels):
    # Concatenate only non-empty arrays
    non_empty_points = [arr for arr in points if arr.size != 0]
    points_all = np.concatenate(non_empty_points, axis=0)
    # points_all = np.concatenate(points, axis=0)
    print(f'points_all shape: {points_all.shape}')
 
    # Convert segmentation labels to a flat NumPy array
    non_empty_labels = [arr for arr in segmentation_labels if arr.size != 0]
    segmentation_labels = np.concatenate(non_empty_labels, axis=0)
    # segmentation_labels = np.concatenate(segmentation_labels, axis=0)
    print(f'segmentation_labels shape: {segmentation_labels.shape}')
 
    show_point_cloud_with_labels(points_all, segmentation_labels)


def show_point_cloud_with_labels(points, segmentation_labels):
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

    # Labels
    # Extract unique class IDs and instance IDs
    unique_instance_ids = np.unique(segmentation_labels[:, 0])
    unique_segment_ids = np.unique(segmentation_labels[:, 1])

    # Create a color mapping based on class IDs
    class_color_mapping = {class_id: plt.cm.tab20(i) for i, class_id in enumerate(unique_segment_ids)}

    # Create a color array based on segmentation labels
    colors = np.array([class_color_mapping[class_id] for class_id in segmentation_labels[:, 1]])


    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    vis.run()

def filter_lidar_data(point_clouds, segmentation_labels):
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
            if (not np.any(label == 0) and (label[1] == 1 or label[1] == 8 or label[1] == 10)):
            # if (not np.any(label == 0)):
                filtered_lidar_point.append(point)
                filtered_lidar_label.append(label)
            else: continue
        filtered_points.append(np.array(filtered_lidar_point))
        filtered_labels.append(np.array(filtered_lidar_label))

    return filtered_points, filtered_labels


if __name__ == "__main__":
    from WaymoParser import *
    # from waymo_pointcloud_parser import *

    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_samples")

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
            
            filtered_point_cloud, filtered_point_labels = filter_lidar_data(points_return1, point_labels)


            visualize_pointcloud_return(filtered_point_cloud, filtered_point_labels)
            # visualize_pointcloud_return(points_return1, point_labels)

