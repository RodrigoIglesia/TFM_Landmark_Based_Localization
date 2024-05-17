import os
import sys
import cv2
from ultralytics.utils.plotting import Annotator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()


# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_semantic_parser import *



if __name__ == "__main__":
    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

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

            result_image_panoptic = cv2.addWeighted(decoded_image, 1, panoptic_label_rgb, 0.5, 0)
            draw_bboxes(camera_bboxes, result_image_panoptic)
            result_image_semantic_original = cv2.addWeighted(decoded_image, 1, semantic_label_original_rgb, 0.5, 0)
            draw_bboxes(camera_bboxes, result_image_semantic_original)
            result_image_semantic = cv2.addWeighted(decoded_image, 1, semantic_label_rgb, 0.5, 0)

            frame_index += 1

