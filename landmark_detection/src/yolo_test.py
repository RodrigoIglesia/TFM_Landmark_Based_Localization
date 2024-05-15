#!/usr/bin/env python3
"""
Main file
"""

import os
import sys
import cv2
import random
import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# Add project src root to python path
# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
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
    results = model.predict(source=image, save=False, classes=[0, 2, 9, 10, 11])
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    for r in results:
        for mask, box in zip(r.masks.xy, r.boxes):
            points = np.int32([mask])
            # cv2.polylines(img, points, True, (255, 0, 0), 1)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(black_image, points, colors[color_number])
    return black_image

def plot_segmentation(segmentation_mask, base_image):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    result_image = cv2.addWeighted(base_image, 1, segmentation_mask  , 0.5, 0)
    cv2.imshow('result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_callback(msg, model):
    rospy.loginfo("Image received")
    # Convert ROS Image message to OpenCV image
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # Detect signs
    segment_mask = yolo_detect(image, model)
    result_image = cv2.addWeighted(image, 1, segment_mask, 0.5, 0)
    rospy.loginfo("detection done")
    # plot_segmentation(segment_mask, image)
    # Image message definition
    detection_msg = Image()
    detection_msg.encoding = "bgr8"  # Set image encoding
    detection_msg.is_bigendian = False
    detection_msg.height = result_image.shape[0]  # Set image height
    detection_msg.width = result_image.shape[1]  # Set image width
    detection_msg.step = result_image.shape[1] * 3
    detection_msg.data = result_image.tobytes()
    detection_msg.header.frame_id = "base_link"
    detection_msg.header.stamp = msg.header.stamp # Maintain image acquisition stamp

    # Result publisher
    detection_publisher = rospy.Publisher("image_detection", Image, queue_size=10)
    detection_publisher.publish(detection_msg)
    rospy.loginfo("detection result published")

    # Save result to rosbag
    rosbag_path = os.path.join(src_dir, f"dataset/camera_images_bags/camera_params_{msg.header.frame_id}.bag")
    with rosbag.Bag(rosbag_path, 'w') as bag:
        bag.write('waymo_CameraProjections', detection_msg, detection_msg.header.stamp)
        rospy.loginfo("detection result saved in ROS bag")

if __name__ == "__main__":
    model = YOLO('models/yolov8n-seg.pt')  # load an official model

    # Initialize ROS node
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.loginfo("Landmark detection node initialized correctly")
    rospy.Subscriber("waymo_Camera", Image, image_callback, callback_args=model)

    
    rospy.spin()

