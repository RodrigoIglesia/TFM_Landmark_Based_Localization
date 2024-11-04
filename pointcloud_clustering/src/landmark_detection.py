#!/usr/bin/env python3
"""
Main file
"""

import os
import sys
import cv2
import numpy as np

import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Import tensorflow before transformers with logs deactivated to avoid printing tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torchvision.transforms as T
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pointcloud_clustering.srv import landmark_detection_srv, landmark_detection_srvResponse
import configparser


# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_2d_parser import *
from waymo_utils.publisher_utils import *

cityscapes_classes = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
cityscapes_palette = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

def predict(model, extractor, image, device):
    """
    :param model: The Segformer model.
    :param extractor: The Segformer feature extractor.
    :param image: The image in RGB format.
    :param device: The compute device.
    Returns:
        labels: The final labels (classes) in h x w format.
    """
    pixel_values = extractor(image, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        logits = model(pixel_values).logits
    # Rescale logits to original image size.
    logits = F.interpolate(
        logits,
        size=image.shape[:2],
        mode='bilinear',
        align_corners=False
    )
    # Get class labels.
    labels = torch.argmax(logits.squeeze(), dim=0)
    return labels

def draw_segmentation_map(labels, palette):
    """
    :param labels: Label array from the model.Should be of shape 
        <height x width>. No channel information required.
    :param palette: List containing color information.
        e.g. [[0, 255, 0], [255, 255, 0]] 
    """
    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    for label_num in range(0, len(palette)):
        index = labels == label_num
        red_map[index] = np.array(palette)[label_num, 0]
        green_map[index] = np.array(palette)[label_num, 1]
        blue_map[index] = np.array(palette)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    """
    :param image: Image in RGB format.
    :param segmented_image: Segmentation map in RGB format. 
    """
    alpha = 0.5 # transparency for the original image
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image


def process_image_service(req):
    rospy.loginfo("Image received for processing")
    # Load the model and feature extractor from Hugging Face
    model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model.to('cpu').eval()


    # Convert ROS Image message to OpenCV image
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(req.image, desired_encoding="bgr8")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # DEBUG publishing -> input image
    publish_image_to_topic(topic='/landmark_detection_input', image=image, header=req.image.header)


    # Get labels.
    labels = predict(model, feature_extractor, image, 'cpu')
    # Get segmentation map.
    seg_map = draw_segmentation_map(
        labels.cpu(), cityscapes_palette
    )
    # DEBUG publishing -> segmentation map
    publish_image_to_topic(topic='/landmark_detection_seg_map', image=seg_map, header=req.image.header)

    outputs = image_overlay(image, seg_map)
    # DEBUG publishing -> output result
    publish_image_to_topic(topic='/landmark_detection_output', image=outputs, header=req.image.header)

    # cv2.imshow('Image', outputs)
    # cv2.waitKey(0)
    # Convert result back to ROS Image message

    # Create response mesage of the service
    detection_msg = bridge.cv2_to_imgmsg(seg_map, encoding="bgr8")
    detection_msg.header = req.image.header  # Maintain image header

    return landmark_detection_srvResponse(processed_image=detection_msg)

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('landmark_detection', anonymous=True)
    rospy.loginfo("Landmark detection server initialized correctly")

    # Create service
    service = rospy.Service('landmark_detection', landmark_detection_srv, process_image_service)
    rospy.loginfo("Service 'landmark_detection' is ready")

    rospy.spin()
