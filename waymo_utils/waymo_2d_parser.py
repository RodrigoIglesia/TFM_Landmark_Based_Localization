"""
Este módulo contiene las funciones necesarias para
 - paresear información del Waymo dataset
 - extraer las imágenes de las cámaras
"""

import os
import sys
import pathlib
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)
print(src_dir)

from waymo_utils.WaymoParser import *


def convert_annot2yolo(annotations, image_height, image_width):
    """
    Converts a given list of annotations (dictionary) in a list of annotations in yolo format.
    Each element of the resulting list is another list [class, center_x, center_y, width, height]
    Args: annotations - list of annotations dicts
    Returns: yolo_annotations - list of yolo annotations
    """

    yolo_annotations = []
    for bbox in annotations:
        label = bbox.type
        bbox_coords = bbox.box
        xc = bbox_coords.center_x / image_width
        yc = bbox_coords.center_y / image_height
        width = bbox_coords.length / image_width
        height = bbox_coords.width / image_height


        yolo_annot = [int(label),xc, yc, width, height]
        yolo_annotations.append(yolo_annot)

    return yolo_annotations



if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "/dataset/final_tests_scene")
    print(dataset_path)

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    frame_idx = 0
    image_idx = 0
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        # Iterate through frames of the scenes
        for frame in load_frame(scene_path):
            print(frame.timestamp_micros)

            # Read the 5 cammera images of the frame
            cameras_images = {}
            cameras_bboxes = {}
            for i, image in enumerate(frame.images):
                decoded_image = get_frame_image(image)
                cameras_images[image.name] = decoded_image[...,::-1]
                cameras_bboxes[image.name] = get_image_bboxes(frame.camera_labels, image.name)
            ordered_images = order_camera(cameras_images, [4, 2, 1, 3, 5])
            ordered_bboxes = order_camera(cameras_bboxes, [4, 2, 1, 3, 5])

            canvas = generate_canvas(ordered_images)

            # Show cameras images
            cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Canvas", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
