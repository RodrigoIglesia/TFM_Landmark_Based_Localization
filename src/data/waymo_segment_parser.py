"""
Waymo Open Dataset - PVPS Datasets
Parsing and dataset generation for labeling and training
Este módulo contiene funciones para:

TODO: Crear una clase con métodos para parsear waymo, generar las etiquetas panópticas y documentarlo
"""

import os
import sys

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import dask.dataframe as dd
import cv2

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
# from waymo_open_dataset import v2
# from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
# from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
# from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics
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

def get_frame_image(frame_image):
    """
    Decodes a single image contained in the frame
    Args: frame_image - image in waymo format
    Return: decoded_image - Numpy decoded image
    """
    decoded_image = tf.image.decode_jpeg(frame_image.image)
    decoded_image = decoded_image.numpy()

    return decoded_image


if __name__ == "__main__":
    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    scene_path = os.path.join(src_dir, "dataset/waymo_samples/train/individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord")

    frames_with_seg = [] # Vector to save frames with segmentation labels
    sequence_id = None
    for frame in load_frame(scene_path):
        # Each frame contains 5 camera images.
        # Save frames which contain CameraSegmentationLabel messages. We assume that
        # if the first image has segmentation labels, all images in this frame will.
        if frame.images[0].camera_segmentation_label.panoptic_label:
            frames_with_seg.append(frame)
            if sequence_id is None:
                sequence_id = frame.images[0].camera_segmentation_label.sequence_id
            # Collect 3 frames for this demo. However, any number can be used in practice.
            if frame.images[0].camera_segmentation_label.sequence_id != sequence_id or len(frames_with_seg) > 2:
                break


    # Organize the segmentation labels in order from left to right for viz later.
    camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                                open_dataset.CameraName.FRONT_LEFT,
                                open_dataset.CameraName.FRONT,
                                open_dataset.CameraName.FRONT_RIGHT,
                                open_dataset.CameraName.SIDE_RIGHT]
    segmentation_protos_ordered = []
    for frame in frames_with_seg:
        segmentation_proto_dict = {image.name : image.camera_segmentation_label for image in frame.images}
        segmentation_protos_ordered.append([segmentation_proto_dict[name] for name in camera_left_to_right_order])

    ###############################################################################
    ## READ PANOPTIC LABELS
    ###############################################################################
    # The dataset provides tracking for instances between cameras and over time.
    # By setting remap_to_global=True, this function will remap the instance IDs in
    # each image so that instances for the same object will have the same ID between
    # different cameras and over time.
    segmentation_protos_flat = sum(segmentation_protos_ordered, [])
    panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
        segmentation_protos_flat, remap_to_global=True
    )

    NUM_CAMERA_FRAMES = 5
    instance_labels_multiframe = []
    for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
        instance_labels = []
        for j in range(NUM_CAMERA_FRAMES):
            semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                panoptic_labels[i + j], panoptic_label_divisor
            )
            instance_labels.append(instance_label)
        instance_labels_multiframe.append(instance_labels)

    def _pad_to_common_shape(label):
        return np.pad(label, [[1280 - label.shape[0], 0], [0, 0], [0, 0]])

    # Pad labels to a common size so that they can be concatenated.
    instance_labels = [[_pad_to_common_shape(label) for label in instance_labels] for instance_labels in instance_labels_multiframe]
    instance_labels = [np.concatenate(label, axis=1) for label in instance_labels]

    instance_label_concat = np.concatenate(instance_labels, axis=0)
    panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
        semantic_label_concat, instance_label_concat)
    
    cv2.namedWindow('panoptic', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('panoptic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("panoptic", instance_label_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()