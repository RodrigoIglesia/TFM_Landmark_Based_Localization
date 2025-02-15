#!/usr/bin/env python3
"""
Main file with optimizations in pre- and post-processing (Option 3)
"""

import os
import sys
import cv2
import numpy as np

# Import tensorflow before transformers with logs deactivated to avoid printing tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

import torch
import torchvision.transforms as T
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pointcloud_clustering.srv import landmark_detection_srv, landmark_detection_srvResponse
import configparser

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.deprecation")

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
    Vectoriza la asignación de colores a cada etiqueta utilizando operaciones de NumPy.
    
    :param labels: Array de etiquetas en formato h x w. Si es un tensor, se convierte a numpy.
    :param palette: Lista con la información de color para cada clase.
    Returns:
        segmentation_map: Imagen de segmentación en formato h x w x 3.
    """
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()
    palette_array = np.array(palette, dtype=np.uint8)
    # Assign vectorized color to each label
    segmentation_map = palette_array[labels]
    return segmentation_map

def image_overlay(image, segmented_image):
    """
    Optimiza la superposición de imágenes realizando conversiones mínimas.
    
    :param image: Imagen original en RGB.
    :param segmented_image: Mapa de segmentación en RGB.
    Returns:
        Imagen resultante en formato BGR para su visualización.
    """
    alpha = 0.5  # transparency for the original image
    beta = 1.0   # trasnsparency for the segmented image
    gamma = 0
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, alpha, segmented_bgr, beta, gamma)
    return overlay

def process_image_service(req, model, feature_extractor):
    rospy.logdebug("Image received for processing")
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(req.image, desired_encoding="bgr8")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # DEBUG publishing -> imagen de entrada
    publish_image_to_topic(topic='/landmark_detection_input', image=image_rgb, header=req.image.header)

    # Prediction phase
    labels = predict(model, feature_extractor, image_rgb, 'cpu')
    seg_map = draw_segmentation_map(labels.cpu(), cityscapes_palette)
    # DEBUG publishing -> mapa de segmentación
    publish_image_to_topic(topic='/landmark_detection_seg_map', image=seg_map, header=req.image.header)

    # Overlay results on the original image
    outputs = image_overlay(image_rgb, seg_map)
    # DEBUG publishing -> imagen con superposición
    publish_image_to_topic(topic='/landmark_detection_output', image=outputs, header=req.image.header)

    # Convert segmented image to ROS message
    detection_msg = bridge.cv2_to_imgmsg(seg_map, encoding="bgr8")
    detection_msg.header = req.image.header

    return landmark_detection_srvResponse(processed_image=detection_msg)

if __name__ == "__main__":
    rospy.init_node('landmark_detection', anonymous=True)
    rospy.logdebug("Landmark detection server initialized correctly")

    # Load segformer model
    model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model.to('cpu').eval()

    # Create ROS service
    service = rospy.Service('landmark_detection', landmark_detection_srv,  lambda req: process_image_service(req, model, feature_extractor))
    rospy.logdebug("Service 'landmark_detection' is ready")

    rospy.spin()
