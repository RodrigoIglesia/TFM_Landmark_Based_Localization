"""
This script parses the waymo dataset.
For each frame, point clous is extracter and published in a rosbag
"""

#!/usr/bin/env python

import os
import sys

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import pathlib

from data import WaymoParser as wp
from data import waymo_pointcloud_parser as wpc

from waymo_open_dataset import dataset_pb2 as open_dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils

def main():
    rospy.init_node('waymo_pointcloud_publisher', anonymous=True)
    pub = rospy.Publisher('/waymo_pointcloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)  # Adjust the rate according to your needs

    # Add project root root to python path
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
    sys.path.append(src_dir)

    dataset_path = os.path.join(src_dir, "dataset/waymo_samples/train")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        # Iterate through frames in the dataset
        for frame in wp.load_frame(scene_path):
            points, points_cp = wpc.frame_to_pointclod(frame)

            # Create PointCloud2 message
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'lidar_frame'

            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]

            pc2_msg = pc2.create_cloud_xyz32(header, fields, points)

            # Publish the PointCloud2 message
            pub.publish(pc2_msg)

            rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
