"""
Waymo Open Dataset - PVPS Datasets
Parsing and dataset generation for labeling and training
Este módulo contiene funciones para:

TODO: Crear una clase con métodos para parsear waymo, generar las etiquetas panópticas y documentarlo
"""

import os
import sys

import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

import immutabledict

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.utils import camera_segmentation_utils

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)


from waymo_utils.WaymoParser import *
from waymo_utils.bbox_utils import *

SIGNS_COLOR_MAP = immutabledict.immutabledict({
    cs_pb2.CameraSegmentation.TYPE_UNDEFINED: [0, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_EGO_VEHICLE: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_CAR: [0, 0, 142],
    cs_pb2.CameraSegmentation.TYPE_TRUCK: [0, 0, 70],
    cs_pb2.CameraSegmentation.TYPE_BUS: [0, 60, 100],
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE: [61, 133, 198],
    cs_pb2.CameraSegmentation.TYPE_BICYCLE: [119, 11, 32],
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE: [0, 0, 230],
    cs_pb2.CameraSegmentation.TYPE_TRAILER: [111, 168, 220],
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN: [220, 20, 60],
    cs_pb2.CameraSegmentation.TYPE_CYCLIST: [255, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST: [180, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_BIRD: [127, 96, 0],
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL: [91, 15, 0],
    cs_pb2.CameraSegmentation.TYPE_CONSTRUCTION_CONE_POLE: [235, 12, 12], #14
    cs_pb2.CameraSegmentation.TYPE_POLE: [110, 245, 69], #15
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT: [69, 122, 245], #16
    cs_pb2.CameraSegmentation.TYPE_SIGN: [245, 69, 245], #17
    cs_pb2.CameraSegmentation.TYPE_TRAFFIC_LIGHT: [239, 245, 69], #18
    cs_pb2.CameraSegmentation.TYPE_BUILDING: [70, 70, 70],
    cs_pb2.CameraSegmentation.TYPE_ROAD: [128, 64, 128],
    cs_pb2.CameraSegmentation.TYPE_LANE_MARKER: [234, 209, 220],
    cs_pb2.CameraSegmentation.TYPE_ROAD_MARKER: [217, 210, 233],
    cs_pb2.CameraSegmentation.TYPE_SIDEWALK: [244, 35, 232],
    cs_pb2.CameraSegmentation.TYPE_VEGETATION: [107, 142, 35],
    cs_pb2.CameraSegmentation.TYPE_SKY: [70, 130, 180],
    cs_pb2.CameraSegmentation.TYPE_GROUND: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_DYNAMIC: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_STATIC: [102, 102, 102],
})


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


def get_semantic_labels(image):
    segmentation_proto = image.camera_segmentation_label
    panoptic_label_divisor = image.camera_segmentation_label.panoptic_label_divisor

    panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(segmentation_proto)

    # Get semantic and instance label from panoptic label
    semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
        panoptic_label, panoptic_label_divisor
    )

    return semantic_label, instance_label


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

def get_class_name(semantic_label, instance_id):
    # Map instance_id to class name based on your semantic labels
    # This is just an example, update it based on your actual dataset
    return "car" if semantic_label[instance_id] == 1 else "pedestrian"


def get_signs_color_map(color_map_dict= None):
    if color_map_dict is None:
        color_map_dict = SIGNS_COLOR_MAP
    classes = list(color_map_dict.keys())
    colors = list(color_map_dict.values())
    color_map = np.zeros([np.amax(classes) + 1, 3], dtype=np.uint8)
    color_map[classes] = colors
    return color_map

def merge_semantic_labels(semantic_mask):
    # Merge semantic_mask with a same color
    semantic_mask_merged = semantic_mask.copy()
    semantic_mask_merged[(semantic_mask_merged != [0, 0, 0]).any(axis=-1)] = [255,0,255]

    return semantic_mask_merged

if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_test_scene")

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
        cameras_images = {}
        cameras_bboxes = {}
        cameras_images_bboxes = {}
        cameras_semantics = {}
        cameras_panoptics = {}
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)
            cameras_images[image.name] = decoded_image[...,::-1]
            bboxes = get_image_bboxes(frame.camera_labels, image.name)
            cameras_bboxes[image.name] = bboxes

            image_bboxes = decoded_image.copy()
            draw_bboxes(bboxes, image_bboxes)
            cameras_images_bboxes[image.name] = image_bboxes
            image_size = (decoded_image.shape[1], decoded_image.shape[0])

            semantic_label, instance_label = get_semantic_labels(image)

            ###########################################################
            # Keep only interest classes in semantic classes
            values_to_keep = [15, 17, 18]
            # Create a copy of the original semantic labels to keep them
            semantic_label_signs = np.copy(semantic_label)
            # Create a mask for the values you want to keep
            mask = np.isin(semantic_label, values_to_keep)
            # Apply the mask to set other pixel values to 0
            semantic_label_signs[mask == False] = 0


            # Get panoptic label and plot on top of the image
            panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
                semantic_label, instance_label
            )
            cameras_panoptics[image.name] = panoptic_label_rgb

            semantic_label_original_rgb = camera_segmentation_utils.semantic_label_to_rgb(semantic_label)

            # Only get segmentation labels of desired objects
            color_map = get_signs_color_map()
            semantic_label_rgb = camera_segmentation_utils.semantic_label_to_rgb(semantic_label_signs, color_map)
            semantic_label_rgb_merged = merge_semantic_labels(semantic_label_rgb)
            cameras_semantics[image.name] = semantic_label_rgb_merged

        # Order images and labels by cameras
        ordered_images = order_camera(cameras_images, [4, 2, 1, 3, 5])
        ordered_bboxes = order_camera(cameras_bboxes, [4, 2, 1, 3, 5])
        ordered_images_bboxes = order_camera(cameras_images_bboxes, [4, 2, 1, 3, 5])
        ordered_semantics = order_camera(cameras_semantics, [4, 2, 1, 3, 5])
        ordered_panoptics = order_camera(cameras_panoptics, [4, 2, 1, 3, 5])

        images_canvas = generate_canvas(ordered_images)
        images_bboxes_canvas = generate_canvas(ordered_images_bboxes)
        semantics_canvas = generate_canvas(ordered_semantics)
        panoptic_canvas = generate_canvas(ordered_panoptics)

        result_image_panoptic = cv2.addWeighted(ordered_images[2], 1, ordered_panoptics[2], 0.5, 0)
        result_image_semantic = cv2.addWeighted(ordered_images[2], 1, ordered_semantics[2], 0.5, 0)
 
        # result_image = cv2.vconcat([images_canvas, result_image_panoptic, result_image_semantic])

        plt.imshow(cv2.cvtColor(result_image_semantic, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # cv2.namedWindow('panoptic', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('panoptic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("panoptic", images_canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        frame_index += 1
