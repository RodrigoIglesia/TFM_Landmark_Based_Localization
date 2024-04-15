"""
    This node reads an point cloud from Waymo Open Dataset, converts it to PCL and publishd to a determined topic
"""
import os
import sys

import logging
import numpy as np

import rospy
# ROS libraries to work with mesates
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *
from waymo_utils.waymo_pointcloud_parser import *

def callback_pointcloud(msg):
    rospy.loginfo("Subscribed to PointCloud topic...")
    # Convert ROS PointCloud2 message to numpy array
    try:
        points = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
    except msg == None:
        rospy.logerr("No message received...")
    except:
        rospy.logerr("Error while converting PointCloud...")
    show_point_cloud(points)


if __name__ == '__main__':
    rospy.init_node('pointcloud_viewer', anonymous=True)
    rospy.loginfo("Node initialized correctly")
    
    rospy.Subscriber('waymo_pointcloud', PointCloud2, callback_pointcloud)
    rospy.spin()