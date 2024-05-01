"""
Main file
"""

import os
import sys
import cv2
import random
from ultralytics import YOLO
from waymo_open_dataset import dataset_pb2 as open_dataset

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
    height, width, _ = image.shape
    # Create a black image of the same dimensions
    black_image = 255 * np.ones((height, width, 3), dtype=np.uint8)

    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    results = model.predict(source=canvas, save=False, classes=[0, 2, 9, 10, 11])
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    for r in results:
        for mask, box in zip(r.masks.xy, r.boxes):
            points = np.int32([mask])
            # cv2.polylines(img, points, True, (255, 0, 0), 1)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(black_image, points, colors[color_number])
    return black_image

if __name__ == "__main__":
    model = YOLO('yolov8n-seg.pt')  # load an official model
    

    cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    detection_mask = yolo_detect(canvas, model)
    result_image = cv2.addWeighted(canvas, 1, detection_mask  , 0.5, 0)

    cv2.imshow('Canvas', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

