import os
import sys
import cv2
import numpy as np

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)


from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *
from src.waymo_utils.waymo_semantic_parser import *
from src.waymo_utils.bbox_utils import *

if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene_mod")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    # Get only frames with segmentation labels and append to a list of frames
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        images = {}
        for frame in load_frame(scene_path):
            # Each frame contains 5 camera images.
            # if the first image has segmentation labels, all images in this frame will.
            if not(frame.images[0].camera_segmentation_label.panoptic_label):
                continue

            ##########################################################<
            # Only get segmentation labels of desired objects
            for i, image in enumerate(frame.images):
                decoded_image = get_frame_image(image)

                # Get segmentation labels from the image
                semantic_label, instance_label = get_semantic_labels(image)
                
                # Only semantic segmentation label is used > no instances of signs.
                # Keep only interest classes in semantic classes
                # 15 -> POLE
                # 17 -> SIGN
                # 18 -> TRAFFIC LIGHT
                values_to_keep = [15, 17, 18]
                # Create a copy of the original semantic labels to keep them
                semantic_label_signs = np.copy(semantic_label)
                # Create a mask for the values you want to keep
                mask = np.isin(semantic_label, values_to_keep)
                # Apply the mask to set other pixel values 0
                semantic_label_signs[mask == False] = 0

                color_map = get_signs_color_map()
                semantic_label_rgb = camera_segmentation_utils.semantic_label_to_rgb(semantic_label_signs, color_map)

                result_image_semantic = cv2.addWeighted(decoded_image, 1, semantic_label_rgb, 0.5, 0)

                # Semantic Objects bbox
                semantic_mask = cv2.cvtColor(semantic_label_rgb, cv2.COLOR_BGR2GRAY)
                _, semantic_mask = cv2.threshold(semantic_mask, 127, 255, cv2.THRESH_BINARY)
                # Find contours in the mask image
                contours, _ = cv2.findContours(semantic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw bounding boxes on the mask image
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_image_semantic, (x, y), (x + w, y + h), (255, 255, 0), 2)
                def _pad_to_common_shape(label):
                    return np.pad(label, [[1280 - label.shape[0], 0], [0, 0], [0, 0]])

                images[image.name] = _pad_to_common_shape(result_image_semantic)

            # Order images with camera names
            camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                              open_dataset.CameraName.FRONT_LEFT,
                              open_dataset.CameraName.FRONT,
                              open_dataset.CameraName.FRONT_RIGHT,
                              open_dataset.CameraName.SIDE_RIGHT]
            
            images_ordered = [images[name] for name in camera_left_to_right_order]

            frame_panoptic = generate_canvas(images_ordered)
            cv2.namedWindow('panoptic', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('panoptic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("panoptic", frame_panoptic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
