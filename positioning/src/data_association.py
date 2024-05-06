#!/usr/bin/env python3
"""
Data association Node
Subscribes to:
    Clustered pointcloud topic
    Image Segmented topic
1. Projects the pointcloud on the segmented image
2. Applies IoU on the pointcloud and segmented image, determining which points correspond to the desired elements.
3. For the filtered pointcloud, obtains the coordinates (referenced to the map) and orientation (roll, pitch, yaw) of each element and publishes it.
The process is sequential: First read the pointcloud, then read the image, then associate the data and finally filter the pointcloud.
"""

import cv2
import ctypes
import struct
import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2, Image
from waymo_parser.msg import CameraProj
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

class DataAssociation:
    def __init__(self):
        self.pointcloud_processed = False
        # Initialize the received pointcloud
        self.pointcloud = np.array([[0,0,0]])
        self.cluster_labels = np.array([[0,0,0]])
        # Initialize Camera projections
        self.camera_intrinsic = np.zeros((9,))
        self.camera_extrinsic = np.zeros((4,4))

    def pointcloud_callback(self, msg):
        rospy.loginfo("pointcloud received")

        # Extract xyz and rgb data and load to numpy array
        gen = pc2.read_points(msg, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            self.pointcloud = np.append(self.pointcloud,[[x[0],x[1],x[2]]], axis = 0)
            self.cluster_labels = np.append(self.cluster_labels,[[r,g,b]], axis = 0)
        rospy.loginfo("Pointcloud decoded")

        # # Visualize labeled pointcloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.pointcloud)
        # pcd.colors = o3d.utility.Vector3dVector(self.cluster_labels / 255.0)  # Normalize colors to range [0, 1]
        # o3d.visualization.draw_geometries([pcd])

    def camera_projections_callback(self, msg):
        rospy.loginfo("Camera Projections received")

        # Reshape camera projections to original shape
        self.camera_intrinsic = msg.intrinsic
        self.camera_extrinsic = np.reshape(msg.extrinsic, (4,4))

    def image_callback(self, msg):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('data_association', anonymous=True)
    rospy.loginfo("Data association detection node initialized correctly")

    da = DataAssociation()

    # First subscribe to clustered pointcloud
    rospy.Subscriber("clustered_PointCloud", PointCloud2, da.pointcloud_callback)
    if da.pointcloud_processed == True:
        # Subscribe to camera parameters
        rospy.Subscriber("waymo_CameraProjections", CameraProj, da.camera_projections_callback)
        # do not read an image until the pointcloud has been processed > avoid 
        rospy.Subscriber("image_detection", Image, da.image_callback)
        da.pointcloud_processed = False # Until another pointcloud is received
    rospy.spin()