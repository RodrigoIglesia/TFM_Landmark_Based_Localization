#!/usr/bin/env python3
"""
Main file
"""

import os
import sys
import cv2
import random
import numpy as np
import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from pointcloud_clustering.srv import landmark_detection_srv, landmark_detection_srvResponse

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_2d_parser import *


def yolo_detect(image, model):
    """
    Runs YOLO detection on the input image and returns the segmentation mask.
    """
    height, width, _ = image.shape
    # Create a black image of the same dimensions
    black_image = 255 * np.ones((height, width, 3), dtype=np.uint8)

    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    results = model.predict(source=image, save=False, classes=[0, 2, 9, 10, 11])
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    for r in results:
        for mask, box in zip(r.masks.xy, r.boxes):
            points = np.int32([mask])
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(black_image, points, colors[color_number])
    return black_image


def process_image_service(req):
    rospy.loginfo("Image received for processing")
    # Convert ROS Image message to OpenCV image
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(req.image, desired_encoding="bgr8")

    # Detect signs
    segment_mask = yolo_detect(image, model)
    result_image = cv2.addWeighted(image, 1, segment_mask, 0.5, 0)
    rospy.loginfo("Detection done")

    # Convert result back to ROS Image message
    detection_msg = bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
    detection_msg.header = req.image.header  # Maintain image header


    return landmark_detection_srvResponse(processed_image=detection_msg)


if __name__ == "__main__":
    model = YOLO('models/yolov8n-seg.pt')  # load an official model

    # Initialize ROS node
    rospy.init_node('landmark_detection', anonymous=True)
    rospy.loginfo("Landmark detection server initialized correctly")

    # Create service
    service = rospy.Service('landmark_detection', landmark_detection_srv, process_image_service)
    rospy.loginfo("Service 'landmark_detection' is ready")

    rospy.spin()
