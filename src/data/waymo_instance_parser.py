"""
Waymo Open Dataset - PVPS Datasets
Parsing and dataset generation for labeling and training
Este módulo contiene funciones para:

TODO: Crear una clase con métodos para parsear waymo, generar las etiquetas panópticas y documentarlo
"""

import os
import sys

import pathlib
import tensorflow as tf
import dask.dataframe as dd
import cv2
import numpy as np

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

def get_frames_with_segment(frames_with_seg, scene):
    """
    Get frames with segmentation labels from an scene
    """
    sequence_id = None
    for frame in load_frame(scene):
        # Each frame contains 5 camera images.
        # Save frames which contain CameraSegmentationLabel messages. We assume that
        # if the first image has segmentation labels, all images in this frame will.
        if frame.images[0].camera_segmentation_label.panoptic_label:
            frames_with_seg.append(frame)
            if sequence_id is None:
                sequence_id = frame.images[0].camera_segmentation_label.sequence_id


def get_frame_image(frame_image):
    """
    Decodes a single image contained in the frame
    Args: frame_image - image in waymo format
    Return: decoded_image - Numpy decoded image
    """
    decoded_image = tf.image.decode_jpeg(frame_image.image)
    decoded_image = decoded_image.numpy()

    return decoded_image

def get_image_bboxes(frame_labels, image_name):
    """
    Parses the frame object and gets the bboxes which corresponds to the desired image
    Args:
        frame_labels - list of labels corresponding to all the images in the frame
        image_name - name of the desired image to label
    Return: bboxes (list of dictionaries)
    """

    bboxes = []
    for camera_labels in frame_labels:
        # Check label name
        if camera_labels.name != image_name:
            continue
        for label in camera_labels.labels:
            bboxes.append(label)

    return bboxes


def convert_to_yolo_format(image_name, frame, panoptic_label_divisor):
    yolo_lines = []

    for i, image in enumerate(frame.images):
        if image.name == image_name:
            segmentation_proto = image.camera_segmentation_label
            panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(segmentation_proto)

            # Get semantic and instance labels from panoptic label
            semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                panoptic_label, panoptic_label_divisor
            )

            # Get the image size
            image_size = (image.image.size[1], image.image.size[0])

            # Convert instance labels to YOLO format
            yolo_lines.extend(convert_instance_labels_to_yolo(semantic_label, instance_label, image_size))

    return yolo_lines

def convert_instance_labels_to_yolo(semantic_label, instance_label, image_size):
    yolo_lines = []

    # Define the YOLO class indices corresponding to your classes
    class_indices = {'car': 0, 'pedestrian': 1}  # Update with your class names and indices

    for instance_id in np.unique(instance_label):
        if instance_id == 0:
            continue  # Skip background

        # Get the bounding box coordinates and class label
        mask = (instance_label == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_center = (x + w / 2) / image_size[1]
            y_center = (y + h / 2) / image_size[0]
            width = w / image_size[1]
            height = h / image_size[0]

            class_label = class_indices.get(get_class_name(semantic_label, instance_id), -1)
            if class_label != -1:
                yolo_lines.append(f"{class_label} {x_center} {y_center} {width} {height}")

    return yolo_lines


if __name__ == "__main__":
    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_samples/train")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    # Get only frames with segmentation labels and append to a list of frames
    frames_with_seg = []
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        get_frames_with_segment(frames_with_seg, scene_path)

    # Read panoptic labels from selected frames
    panoptic_labels = []
    frame_index = 0
    for frame in frames_with_seg:
        # Read the 5 cammera images of the frame
        camearas_images = []
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)
            image_size = (decoded_image.shape[1], decoded_image.shape[0])
            camera_bboxes = get_image_bboxes(frame.camera_labels, image.name)

            segmentation_proto = image.camera_segmentation_label
            panoptic_label_divisor = image.camera_segmentation_label.panoptic_label_divisor

            panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(segmentation_proto)

            # Get semantic and instance label from panoptic label
            semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                panoptic_label, panoptic_label_divisor
            )

            panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
                semantic_label, instance_label
            )

            # Convert instance label to YOLO format
            # yolo_format = instance_label_to_yolo_format(instance_label, image_size)
            # print(f"Frame {frame_index}, Image {i} - YOLO Format: {yolo_format}")

            result_image = cv2.addWeighted(decoded_image, 1, panoptic_label_rgb, 0.5, 0)

            # Print bounding box on top of the panoptic labels
            for bbox in camera_bboxes:
                label = bbox.type
                bbox_coords = bbox.box
                br_x = int(bbox_coords.center_x + bbox_coords.length/2)
                br_y = int(bbox_coords.center_y + bbox_coords.width/2)
                tl_x = int(bbox_coords.center_x - bbox_coords.length/2)
                tl_y = int(bbox_coords.center_y - bbox_coords.width/2)

                tl = (tl_x, tl_y)
                br = (br_x, br_y)

                result_image = cv2.rectangle(result_image, tl, br, (255,0,0), 2)

            cv2.namedWindow('panoptic', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('panoptic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("panoptic", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            frame_index += 1
