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
import matplotlib.pyplot as plt
from matplotlib import cm
import rospy
from sensor_msgs.msg import PointCloud2, Image
from waymo_parser.msg import CameraProj
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from collections import deque

class DataAssociation:
    def __init__(self):
        # Initialize the received pointcloud
        self.pointcloud_queue = deque()
        # Initialize Camera projections
        self.intrinsic_matrix_queue = deque()
        self.extrinsic_matrix_queue = deque()
        self.extrinsic_matrix_inv_queue = deque()
        # Initialize image
        self.image_queue = deque()
        self.image_height = 0
        self.image_width = 0

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def pointcloud_callback(self, msg):
        rospy.loginfo("pointcloud received")
        pointcloud = np.array([[0,0,0]])

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
            pointcloud = np.append(pointcloud,[[x[0],x[1],x[2]]], axis = 0)
        self.pointcloud_queue.append(pointcloud)
        rospy.loginfo("Pointcloud decoded")

        # # Visualize labeled pointcloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.pointcloud)
        # pcd.colors = o3d.utility.Vector3dVector(self.cluster_labels / 255.0)  # Normalize colors to range [0, 1]
        # o3d.visualization.draw_geometries([pcd])

    def camera_projections_callback(self, msg):
        rospy.loginfo("Camera Projections received")

        # Reshape camera projections to original shape
        intrinsic_params = msg.intrinsic
        extrinsic_params = msg.extrinsic

        # Convert params to matrix
        intrinsic_matrix = [[intrinsic_params[0], 0, intrinsic_params[2]],
                            [0, intrinsic_params[1], intrinsic_params[3]],
                            [0, 0, 1]]
        self.intrinsic_matrix_queue.append(intrinsic_matrix)
        extrinsic_matrix = np.array(extrinsic_params).reshape(4,4)
        self.extrinsic_matrix_queue.append(extrinsic_matrix)
        # Extract rotation and translation components from the extrinsic matrix
        rotation = extrinsic_matrix[:3, :3]
        translation = extrinsic_matrix[:3, 3]
        # Invert the rotation matrix
        rotation_inv = np.linalg.inv(rotation)
        # Invert the translation
        translation_inv = -np.dot(rotation_inv, translation)
        # Construct the new extrinsic matrix (from camera to vehicle)
        extrinsic_matrix_inv = np.zeros((4, 4), dtype=np.float32)
        extrinsic_matrix_inv[:3, :3] = rotation_inv
        extrinsic_matrix_inv[:3, 3] = translation_inv
        extrinsic_matrix_inv[3, 3] = 1.0
        extrinsic_matrix_inv = extrinsic_matrix_inv[:3,:]
        self.extrinsic_matrix_inv_queue.append(extrinsic_matrix_inv)


    def image_callback(self, msg):
        rospy.loginfo("Image processed received")
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.image_queue.append(image)
        self.image_height, self.image_width, _ = image.shape

        self.associate_image_lidar()
    
    def associate_image_lidar(self):
        # Dequeue parameters by oldest
        if bool(self.pointcloud_queue) == False:
            rospy.loginfo("Pointcloud queue is empty. Passing...")
            return
        rospy.loginfo("Get queues values")
        pointcloud = self.pointcloud_queue.popleft()
        extrinsic_matrix_inv = self.extrinsic_matrix_inv_queue.popleft()
        intrinsic_matrix = self.intrinsic_matrix_queue.popleft()
        image = self.image_queue.popleft()
        # Project Pointcloud on image
        # Project 3D points in vehicle reference to 3D points in camera reference using extrinsic params
        point_cloud_hom = self.cart2hom(pointcloud)  # nx4
        point_cloud_cam =  np.dot(point_cloud_hom, np.transpose(extrinsic_matrix_inv))

        ## Rotate point cloud to match pinhole model axis (Z pointing to the front of the car, Y pointing down ans X pointing to the right)
        theta_x = np.pi/2
        theta_y = -np.pi/2
        theta_z = 0
        # Define rotation matrices around X, Y, and Z axes
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
        # Rotate the point cloud 90 degrees in X
        point_cloud_cam = np.dot(Rx, point_cloud_cam.T).T
        # Rotate the point cloud -90 degrees in Y
        point_cloud_cam = np.dot(Ry, point_cloud_cam.T).T
        # We only have to project in the image the points that are in front of the car > remove points with z<0
        point_cloud_cam = point_cloud_cam[point_cloud_cam[:, 2] >= 0]
        # Project 3D points in camera reference in 2D points in image reference using intrinsic params
        point_cloud_cam_hom = self.cart2hom(point_cloud_cam)
        # Compute perspective projection matrix
        P = np.hstack((intrinsic_matrix, np.zeros((3, 1))))
        point_cloud_image = np.dot(P, point_cloud_cam_hom.T).T
        point_cloud_image = point_cloud_image[:, :2] / point_cloud_image[:, 2:]
        rospy.loginfo("Pointcloud projected from 3D vehicle frame to 3D camera frame")

        # Filtered 2D points > remove points out of the image FOV
        filtered_indices = (
            (point_cloud_image[:, 0] >= 0) & (point_cloud_image[:, 0] < self.image_width) &
            (point_cloud_image[:, 1] >= 0) & (point_cloud_image[:, 1] < self.image_height)
        )
        min_depth = np.min(point_cloud_cam[:, 2])
        max_depth = np.max(point_cloud_cam[:, 2])
        normalized_depth = (point_cloud_cam[:, 2] - min_depth) / (max_depth - min_depth)
        cmap = cm.get_cmap('jet')  # You can choose any colormap you prefer
        colors = cmap(normalized_depth)
        point_cloud_image = point_cloud_image[filtered_indices]
        colors = colors[filtered_indices]
        rospy.loginfo("Pointcloud projected from 3D camera frame to 2D image frame")

        # Publish results for RVIZ visualization
        # detection_association_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # circle_color = (255, 0, 0)
        # circle_radius = 3
        # for point in point_cloud_image:
        #     x, y = int(point[0]), int(point[1])
        #     cv2.circle(detection_association_image, (x, y), circle_radius, circle_color, -1)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.scatter(point_cloud_image[:, 0], point_cloud_image[:, 1], color=colors, s=5)
        plt.colorbar(label='Depth')
        plt.show()

        # detection_association_msg = Image()
        # detection_association_msg.encoding = "bgr8"  # Set image encoding
        # detection_association_msg.is_bigendian = False
        # detection_association_msg.height    = detection_association_image.shape[0]  # Set image height
        # detection_association_msg.width     = detection_association_image.shape[1]  # Set image width
        # detection_association_msg.step      = detection_association_image.shape[1] * 3
        # detection_association_msg.data      = detection_association_image.tobytes()
        # detection_association_msg.header.frame_id = "base_link"
        # detection_association_msg.header.stamp = rospy.Time.now() # Maintain image acquisition stamp

        # detection_association_publisher = rospy.Publisher("detection_association", Image, queue_size=10)
        # detection_association_publisher.publish(detection_association_msg)


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('data_association', anonymous=True)
    rospy.loginfo("Data association detection node initialized correctly")

    da = DataAssociation()

    # Subscribe to clustered pointcloud
    rospy.Subscriber("clustered_PointCloud", PointCloud2, da.pointcloud_callback)
    # Subscribe to camera parameters
    rospy.Subscriber("waymo_CameraProjections", CameraProj, da.camera_projections_callback)
    # Subscribe to image callback
    rospy.Subscriber("image_detection", Image, da.image_callback)

    rospy.spin()