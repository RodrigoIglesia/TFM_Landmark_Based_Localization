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
# ROS libraries to work with messages
from sensor_msgs.msg import PointCloud2, PointField
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


def pointcloud_to_ros(points, frame_id):
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16  # Adjust point step based on number of fields
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    # Concatenate points and intensities if 'intensity' is provided
    data = np.column_stack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
    msg.data = data.tostring()
    return msg


if __name__ == "__main__":
    rospy.init_node('waymo_pointcloud_publisher', anonymous=True)
    rospy.loginfo("Node initialized correctly")

    pointcloud_publisher = rospy.Publisher('waymo_pointcloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)  # Adjust the publishing rate as needed

    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")
    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    ##############################################################
    ## Iterate through Dataset Scenes
    ##############################################################
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        rospy.loginfo("Scene {} processing: {}".format(str(scene_index), scene_path))

        ##############################################################
        ## Get Scene Point Cloud
        ##############################################################
        # If map features were found, parse the 3D point clouds in the frames
        # Array to store segmented pointclouds
        point_clouds = []
        # Array to store pointcloud labels
        point_cloud_labels = []
        frame_idx = 0
        for frame in load_frame(scene_path):
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

            # Convert concatenated point cloud to ROS message
            pointcloud_msg = pointcloud_to_ros(points, frame_id)

            pointcloud_publisher.publish(pointcloud_msg)
            rospy.loginfo("Concatenated point cloud published")
            
            rate.sleep()

            frame_idx += 1