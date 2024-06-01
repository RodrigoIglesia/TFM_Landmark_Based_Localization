import os
import sys
import pathlib
import numpy as np
import cv2

from roboflow import Roboflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)

from waymo_utils.waymo_semantic_parser import *
from waymo_utils.WaymoParser import *
from waymo_utils.bbox_utils import *

color_to_class = {
    (235, 12, 12): 1,  # Construction cone
    (110, 245, 69): 2,  # Pole
    (69, 122, 245): 3,  # Pedestrian object
    (245, 69, 245): 4,  # Sign
    (239, 245, 69): 5,  # Traffic light
}

def get_yolo_format_coordinates(coords, img_shape):
    """Convert pixel coordinates to normalized coordinates for YOLO format."""
    height, width, _ = img_shape
    return [(x / width, y / height) for x, y in coords]

def find_contours(mask):
    """Find contours in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def convert_to_yolo_format(label_array):
    yolo_data = []
    for color, class_index in color_to_class.items():
        mask = np.all(label_array == color, axis=-1).astype(np.uint8)
        contours = find_contours(mask)
        
        for contour in contours:
            if len(contour) >= 3:  # YOLO format requires at least 3 points to form a valid polygon
                coords = contour.squeeze().tolist()
                yolo_coords = get_yolo_format_coordinates(coords, label_array.shape)
                yolo_entry = [class_index] + [coord for point in yolo_coords for coord in point]
                yolo_data.append(yolo_entry)
    return yolo_data

if __name__ == "__main__":
    ## ROBOFLOW
    # Roboflow API key and project information
    API_KEY = '2oz2py478IAE9Y89tG6q'
    PROJECT_NAME = 'waymo_landmarks'
    WORKSPACE_NAME = 'tfmrodrigo'
    # Initialize Roboflow
    rf = Roboflow(api_key=API_KEY)

    workspace = rf.workspace()
    print(workspace)

    project = workspace.project(PROJECT_NAME)
    print(project)

    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    # Get only frames with segmentation labels and append to a list of frames
    frames_with_seg = []
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        get_frames_with_segment(frames_with_seg, scene_path)

    # Read panoptic labels from selected frames
    panoptic_labels = []
    for frame_index, frame in enumerate(frames_with_seg):
        # Read the 5 camera images of the frame
        camearas_images = []
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)
            image_size = (decoded_image.shape[1], decoded_image.shape[0])
            camera_bboxes = get_image_bboxes(frame.camera_labels, image.name)

            semantic_label, instance_label = get_semantic_labels(image)

            ###########################################################<
            # Keep only interest classes in semantic classes
            values_to_keep = [15, 16, 17, 18, 19]
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

            semantic_label_original_rgb = camera_segmentation_utils.semantic_label_to_rgb(semantic_label)

            # Only get segmentation labels of desired objects
            color_map = get_signs_color_map()
            semantic_label_rgb = camera_segmentation_utils.semantic_label_to_rgb(semantic_label_signs, color_map)

            ## Create Dataset
            # Convert images to YOLO format
            yolo_annotations = convert_to_yolo_format(semantic_label_rgb)
            print("YOLO label generated")

            # Save image and annotation
            print("Saving temp files")
            image_path = f"/home/rodrigo/catkin_ws/src/TFM_Landmark_Based_Localization/landmark_detection/yolo/image_{frame_index}_{i}.jpg"
            cv2.imwrite(image_path, decoded_image)
            annot_path = f"/home/rodrigo/catkin_ws/src/TFM_Landmark_Based_Localization/landmark_detection/yolo/annotation_{frame_index}_{i}.txt"

            with open(annot_path, 'w') as f:
                for line in yolo_annotations:
                    line_str = " ".join(map(str, line))
                    f.write(f"{line_str}\n")
            project.upload(image_path, annot_path)
            print("Instance uploaded to Roboflow")

            # Delete image and annotation files after uploading
            os.remove(image_path)
            os.remove(annot_path)
            print(f"Deleted temp files: {image_path}, {annot_path}")
