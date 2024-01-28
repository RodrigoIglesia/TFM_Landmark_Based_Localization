"""
Este módulo contiene las funciones necesarias para
 - paresear información del Waymo dataset
 - extraer las imágenes de las cámaras
"""

import os
import sys
import pathlib
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

def convert_annot2yolo(annotations):
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

        yolo_annot = [int(label), int(bbox_coords.center_x), int(bbox_coords.center_y), int(bbox_coords.width), int(bbox_coords.length)]
        yolo_annotations.append(yolo_annot)

    return yolo_annotations

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

    dataset_path = os.path.join(src_dir, "dataset/waymo_samples/val")
    images_path = os.path.join(src_dir, "dataset/waymo_yolo_format/images")
    labels_path = os.path.join(src_dir, "dataset/waymo_yolo_format/labels")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    frame_idx = 0
    image_idx = 0
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        # Iterate through frames of the scenes
        for frame in load_frame(scene_path):
            print(frame.timestamp_micros)

            # Read the 5 cammera images of the frame
            camearas_images = []
            for i, image in enumerate(frame.images):
                decoded_image = get_frame_image(image)
                camera_bboxes = get_image_bboxes(frame.camera_labels, image.name)
                camera_yolo_bboxes = convert_annot2yolo(camera_bboxes)

                # Save image
                cv2.imwrite(os.path.join(images_path, f'{frame_idx:05d}_{image_idx:05d}_camera{str(image.name)}.jpg'), cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB) )
                with open(os.path.join(labels_path, f'{frame_idx:05d}_{image_idx:05d}_camera{str(image.name)}.txt'), mode='w') as f:
                    for annot in camera_yolo_bboxes:
                        f.write(' '.join(map(str, annot)))
                        f.write('\n')

                print("Saved " + str(frame.context.name) + "_camera" + str(image.name) + "\n")

                image_idx += 1
            frame_idx += 1
                
                # labeled_image = np.copy(decoded_image)

                # for bbox in camera_bboxes:
                #     label = bbox.type
                #     bbox_coords = bbox.box
                #     br_x = int(bbox_coords.center_x + bbox_coords.length/2)
                #     br_y = int(bbox_coords.center_y + bbox_coords.width/2)
                #     tl_x = int(bbox_coords.center_x - bbox_coords.length/2)
                #     tl_y = int(bbox_coords.center_y - bbox_coords.width/2)

                #     tl = (tl_x, tl_y)
                #     br = (br_x, br_y)

                #     labeled_image = cv2.rectangle(labeled_image, tl, br, (255,0,0), 2)

                # cv2.imshow(str(frame.context.name) + "_camera" + str(image.name), labeled_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()


                # camearas_images.append(decoded_image[...,::-1])
            # canvas = generate_canvas(camearas_images)

            # # Show cameras images
            # cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("Canvas", canvas)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
