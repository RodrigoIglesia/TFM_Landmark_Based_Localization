#!/usr/bin/env python

import rospy
import tf
import math
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import pcl
from pcl import PointCloud, PointIndices
from pcl import NormalEstimation, SACSegmentationFromNormals, ExtractIndices, EuclideanClusterExtraction
from pcl import search
from tf.transformations import quaternion_from_euler

def callback(input):
    global transform
    global msg_ve, msg_ve_BL, msg_ds, msg_cl, msg_ng, msg_ground, marker_array

    clusters = pcl.PointCloud()
    msg_aux = PoseArray()
    msg_aux_BL = PoseArray()
    marker_array_aux = MarkerArray()

    cloud = pcl.PointCloud_PointXYZ()
    pcl.fromROSMsg(input, cloud)

    # Your existing code goes here, but translated into Python

def main():
    global transform
    global msg_ve, msg_ve_BL, msg_ds, msg_cl, msg_ng, msg_ground, marker_array

    rospy.init_node('euclidean_clustering', anonymous=True)
    subscription = rospy.get_param("~subscription")
    sub = rospy.Subscriber(subscription, PointCloud2, callback)
    pub_ds = rospy.Publisher('PointCloud2_ds', PointCloud2, queue_size=1)   # debugging
    pub_ground = rospy.Publisher('poseGround', PoseStamped, queue_size=1)   # debugging
    pub_ng = rospy.Publisher('PointCloud2_ng', PointCloud2, queue_size=1)   # debugging
    pub_cl = rospy.Publisher('PointCloud2_cl', PointCloud2, queue_size=1)   # debugging
    pub_ve = rospy.Publisher('PoseArray_ve', PoseArray, queue_size=1)
    pub_ve_BL = rospy.Publisher('PoseArray_ve_BL', PoseArray, queue_size=1) # As seen from base_link
    pub_text = rospy.Publisher('MarkerArray_text', MarkerArray, queue_size=1) # debugging

    marker_main = Marker()
    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        rospy.spinOnce()
        try:
            (trans,rot) = listener.lookupTransform("map", "base_link", rospy.Time(0))
            transform = tf.TransformerROS(True, rospy.Duration(10.0))
            transform.setTransform((trans, rot), rospy.Time.now(), "base_link", "map")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        pub_ve.publish(msg_ve)
        pub_ve_BL.publish(msg_ve_BL)
        pub_ds.publish(msg_ds)         # debugging
        pub_cl.publish(msg_cl)         # debugging
        pub_ng.publish(msg_ng)         # debugging
        pub_ground.publish(msg_ground) # debugging
        pub_text.publish(marker_array) # debugging

        marker_main.action = Marker.DELETEALL # To avoid dead displays on screen
        marker_array.markers.append(marker_main)
        pub_text.publish(marker_array) # debugging

        rospy.sleep(0.1)

if __name__ == '__main__':
    main()
