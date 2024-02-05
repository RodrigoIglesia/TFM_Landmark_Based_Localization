"""
Waymo Open Dataset - 3D Semantic Segmentation parser (LiDAR)
Este módulo contiene funciones para:
Leer cada frame del dataset y extraer 

"""

import os
import sys

import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
import cv2
import numpy as np

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.utils import camera_segmentation_utils


def load_frame(scene):
    """
    Load and yields frame object of a determined scene
    A frame is composed of imagrs from the 5 cameras
    A frame also has information of the bounding boxes and labels, related to each image
    Args: scene (str) - path to the scene which contains the frames
    Yield: frame_object (dict) - frame object from waymo dataset containing cameras and laser info
    """
    dataset = tf.data.TFRecordDataset(scene, compression_type='')
    for data in dataset:
        frame_object = open_dataset.Frame()
        frame_object.ParseFromString(bytearray(data.numpy()))

        yield frame_object

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


def plot_point_cloud_labels(points, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels[:, 0], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


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

def get_semseg_label_image(laser_name, return_index):
    """Returns semseg label image given a laser name and its return index."""
    return segmentation_labels[laser_name][return_index]

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

if __name__ == "__main__":
    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_samples/train")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        for frame in load_frame(scene_path):
            print(frame.timestamp_micros)

            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            if not(segmentation_labels):
                continue

            # Get points labeled for first and second return
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose)
            points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
            point_labels = convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels)
            point_labels_ri2 = convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels, ri_index=1)

            # 3d points in vehicle frame.
            points_all = np.concatenate(points, axis=0)
            points_all_ri2 = np.concatenate(points_ri2, axis=0)
            # point labels.
            point_labels_all = np.concatenate(point_labels, axis=0)
            point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
            # camera projection corresponding to each point.
            cp_points_all = np.concatenate(cp_points, axis=0)
            cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

            #######################################################
            ## Plot segmentation labels on range images
            ######################################################
            plt.figure(figsize=(64, 20))
            frame.lasers.sort(key=lambda laser: laser.name)
            show_semseg_label_image(get_semseg_label_image(open_dataset.LaserName.TOP, 0), 1)
            plt.show()
            show_semseg_label_image(get_semseg_label_image(open_dataset.LaserName.TOP, 1), 3)
            plt.show()

            #########################################################
            ## Plot 3D segmentation labels on projected image
            #########################################################
            plot_point_cloud_labels(points_all, point_labels_all)
