#!/usr/bin/env python3

"""
    This node reads an point cloud from Waymo Open Dataset, converts it to PCL and publishd to a determined topic
"""
import os
import sys

import logging
import numpy as np
import pathlib

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()


# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *
from waymo_utils.waymo_pointcloud_parser import *

class WaymoPublisher:
    def __init__(self, cams_order):
        rospy.loginfo("Initializing publisher parameters")
        # Vector of integers determining the index order of the cameras, this vector also selects the number of cameras to use
        self.cams_order = cams_order
        # ROS publisher objects
        self.pointcloud_publisher = rospy.Publisher("waymo_PointCloud", PointCloud2, queue_size=10)
        self.camera_publisher = rospy.Publisher("waymo_Camera", Image, queue_size=10)
        # PointCloud2 message definition
        self.pointcloud_msg = PointCloud2()
        self.pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        self.pointcloud_msg.is_bigendian = False
        self.pointcloud_msg.point_step = 16  # Adjust point step based on number of fields
        self.pointcloud_msg.height = 1
        self.pointcloud_msg.is_dense = True
        # Image message definition
        self.camera_msg = Image()
        self.camera_msg.encoding = "bgr8"  # Set image encoding
        self.camera_msg.is_bigendian = False

    def get_pointcloud(self, frame):
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        # Get points labeled for first and second return
        # Parse range image for lidar 1
        def _range_image_to_pcd(ri_index = 0):
            points, points_cp = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose,
                ri_index=ri_index)
            return points, points_cp
        
        # Return of the first 2 lidar scans
        points_return1 = _range_image_to_pcd()
        points_return2 = _range_image_to_pcd(1)

        points, _ = concatenate_pcd_returns(points_return1, points_return2)

        return points
    
    def pointcloud_to_ros(self, points):
        self.pointcloud_msg.width = points.shape[0]
        self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * self.pointcloud_msg.width
        
        # Concatenate points and intensities if 'intensity' is provided
        data = np.column_stack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
        self.pointcloud_msg.data = data.tobytes()

    def publish_pointcloud(self, frame):
        points = self.get_pointcloud(frame)
        if points is not(None):
            # Convert concatenated point cloud to ROS message
            self.pointcloud_to_ros(points)
            # Publish message
            self.pointcloud_publisher.publish(self.pointcloud_msg)
            rospy.loginfo("Concatenated point cloud published")
        else:
            rospy.loginfo("No poincloud in frame")

    def get_camera_image(self, frame):
        # Gets and orders camera images from left to right
        cameras_images = {}
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)
            cameras_images[image.name] = decoded_image[...,::-1]
        # # Create a list of tuples (key, image_data)
        # key_image_tuples = [(key, cameras_images[key]) for key in cameras_images]
        # # Sort the list of tuples based on the custom order
        # sorted_tuples = sorted(key_image_tuples, key=lambda x: self.cams_order.index(x[0]))
        # # Extract image data from sorted tuples
        # ordered_images = [item[1] for item in sorted_tuples]
        ordered_images = [cameras_images[1]]

        return ordered_images

    def publish_camera(self, frame):
        camera_images = self.get_camera_image(frame)
        rospy.loginfo("images loaded from dataset")
        # Check wether 1 or more images has been selected
        if len(camera_images) == 1:
            image = camera_images[0]
            self.camera_msg.height = image.shape[0]  # Set image height
            self.camera_msg.width = image.shape[1]  # Set image width
            self.camera_msg.step = image.shape[1] * 3
            self.camera_msg.data = image.tobytes()

            # Publish message
            self.camera_publisher.publish(self.camera_msg)
            rospy.loginfo("Camera image published")
        elif len(camera_images) > 1:
            # TODO: Implementar stitching de imágenes o publicar una a una.
            pass
        else:
            rospy.loginfo("No image in frame")


if __name__ == "__main__":
    rospy.init_node('waymo_publisher', anonymous=True)
    rospy.loginfo("Node initialized correctly")

    #TODO: Topic configurable
    rate = rospy.Rate(10)  # Adjust the publishing rate as needed
    # TODO: ruta al dataset tiene que ser configurable y única para todos los scripts
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")
    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    # TODO: Cambiar [2] `por una variable configurable que indique la configuración de las cámaras`
    wp = WaymoPublisher([2, 1, 3])

    ##############################################################
    ## Iterate through Dataset Scenes
    ##############################################################
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        rospy.loginfo("Scene {} processing: {}".format(str(scene_index), scene_path))

        frame = next(load_frame(scene_path))
        if frame is not(None):
        # for frame in load_frame(scene_path):
            # TODO: Paralelizar procesos de publicación

            ##################################################
            # Publish LiDAR pointcloud
            ##################################################
            wp.pointcloud_msg.header.frame_id = "base_link"
            wp.pointcloud_msg.header.stamp = rospy.Time.now()
            wp.publish_pointcloud(frame)
            ##################################################
            # Publish Camera Image
            ##################################################
            wp.camera_msg.header.frame_id = "base_link"
            wp.camera_msg.header.stamp = rospy.Time.now()
            wp.publish_camera(frame)
            

            rate.sleep()
