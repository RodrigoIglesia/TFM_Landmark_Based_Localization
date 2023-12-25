"""
Este módulo contiene las funciones necesarias para
 - paresear información del Waymo dataset
 - extraer las imágenes de las cámaras
"""

import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
import cv2

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


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

def get_image_bboxes(image):
    """
    Parses the frame object and gets the bboxes which corresponds to the desired image
    Return: bboxes (list of dictionaries)
    """
    return bboxes

def draw_bboxes(bboxes):
    pass


def generate_canvas(images):
    max_height = max(image.shape[0] for image in images)
    width, _ = images[0].shape[1], images[0].shape[2]
    canvas = np.zeros((max_height, 5 * width, 3), dtype=np.uint8)

    for i in range(5):
        image = images[i]
        height = image.shape[0]

        # Calculate padding values
        top_pad = (max_height - height) // 2
        bottom_pad = max_height - height - top_pad

        # Pad image with black pixels
        padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        canvas[:, i * width:(i + 1) * width, :] = padded_image

    return canvas

if __name__ == "__main__":
    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_samples")
    file = "individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"

    scene_path = os.path.join(dataset_path, file)
    for frame in load_frame(scene_path):
        print(frame.timestamp_micros)

        # Read the 5 cammera images of the frame
        camearas_images = []
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)

            camearas_images.append(decoded_image[...,::-1])

        canvas = generate_canvas(camearas_images)

        # Show cameras images
        cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Canvas", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
