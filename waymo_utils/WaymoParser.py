import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import transform_utils


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

def generate_canvas(images):
    """
    Generateas a canvas of concatenated images for visualization porpuses
    Arg:
        images - list: list of images to concatenatenate
    Returns:
        canvas - numpy ndarray: Generated canvas of concatenated images
    """
    max_height = max(image.shape[0] for image in images)
    width, _ = images[0].shape[1], images[0].shape[2]
    canvas = np.zeros((max_height, len(images) * width, 3), dtype=np.uint8)

    for i in range(len(images)):
        image = images[i]
        height = image.shape[0]

        # Calculate padding values
        top_pad = (max_height - height) // 2
        bottom_pad = max_height - height - top_pad

        # Pad image with black pixels
        padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        canvas[:, i * width:(i + 1) * width, :] = padded_image

    return canvas

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