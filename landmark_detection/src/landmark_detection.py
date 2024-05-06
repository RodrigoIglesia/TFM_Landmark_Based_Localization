#!/usr/bin/env python3
"""
Main file
"""

import os
import sys
import cv2
import random
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_2d_parser import *

class LandmarkDetection:
    def __init__(self):
        self.image_topic = "waymo_Camera"
        self.model = YOLO('models/yolov8n-seg.pt')
        self.image = cv2.Mat()
        self.segment_mask = cv2.Mat()

    def yolo_detect(self, image):
        """
        Returns
        """
        height, width, _ = image.shape
        # Create a black image of the same dimensions
        black_image = 255 * np.ones((height, width, 3), dtype=np.uint8)

        yolo_classes = list(self.model.names.values())
        classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
        results = self.model.predict(source=image, save=False, classes=[0, 2, 9, 10, 11])
        colors = [random.choices(range(256), k=3) for _ in classes_ids]
        for r in results:
            for mask, box in zip(r.masks.xy, r.boxes):
                points = np.int32([mask])
                # cv2.polylines(img, points, True, (255, 0, 0), 1)
                color_number = classes_ids.index(int(box.cls[0]))
                cv2.fillPoly(black_image, points, colors[color_number])
        return black_image

    def image_callback(self, msg):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Detect signs
        self.segment_mask = self.yolo_detect(image)
        rospy.loginfo("detection done")

    def subscribe_image(self):
        rospy.Subscriber(self.image_topic, Image, self.image_callback)


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.loginfo("Landmark detection node initialized correctly")

    # Launch image detector
    ld = LandmarkDetection()
    ld.subscribe_image()

    # Once an image has been processed, obtain the camera parameters

    # Also, obtain the corresponding clustered pointcloud

    rospy.spin()