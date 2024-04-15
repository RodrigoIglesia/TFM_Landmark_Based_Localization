"""
Main file
"""

import os
import sys
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Add project src root to python path
# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.insert(0, src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_2d_parser import *

def yolo_detect(image, model):
    """
    Returns
    """
    annotator = Annotator(image)

    results = model(image)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
          
    annotated_image = annotator.result()

    return annotated_image


if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_samples")
    file = "individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"

    model = YOLO('models/yolov8s.pt')  # load an official model

    scene_path = os.path.join(dataset_path, file)
    for frame in load_frame(scene_path):
        print(frame.timestamp_micros)

        # Read the 5 cammera images of the frame
        camearas_images = []
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)

            result_image = yolo_detect(decoded_image, model)
            # draw_detections(decoded_image, yolo_detections)

            camearas_images.append(result_image[...,::-1])

        canvas = generate_canvas(camearas_images)

        # Show cameras images
        cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Canvas", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()