#!/usr/bin/env python3
"""
Data Association Node
Reads from:
    PointCloud2, Image, and CameraProj rosbags.
Processes:
    Projects the pointcloud on the segmented image, applies IoU on the pointcloud and segmented image, and publishes filtered pointcloud with coordinates and orientation.
"""

import os
import sys

import cv2
import open3d as o3d
import struct
import ctypes
import numpy as np
import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image
from waymo_parser.msg import CameraProj
import sensor_msgs.point_cloud2 as pc2

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)


def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def process_pointcloud(msg):
    pointcloud = np.array([[0,0,0]])
    cluster_labels = np.array([[0,0,0]])

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
        cluster_labels = np.append(cluster_labels,[[r,g,b]], axis = 0)

    return pointcloud, cluster_labels

def process_camera(msg):
    intrinsic_matrix = np.zeros((3,3))
    extrinsic_matrix = np.zeros((4,4))
    extrinsic_matrix_inv = np.zeros((4,4))
    # Reshape camera projections to original shape
    intrinsic_params = msg.intrinsic
    extrinsic_params = msg.extrinsic

    # Convert params to matrix
    intrinsic_matrix = [[intrinsic_params[0], 0, intrinsic_params[2]],
                        [0, intrinsic_params[1], intrinsic_params[3]],
                        [0, 0, 1]]
    extrinsic_matrix = np.array(extrinsic_params).reshape(4,4)
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

    return intrinsic_matrix, extrinsic_matrix, extrinsic_matrix_inv

def process_image(msg):
    rospy.loginfo("Image processed received")
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    image_height, image_width, _ = image.shape

    return image, image_height, image_width

def associate_data(pointcloud, pointcloud_label, intrinsic_matrix, extrinsic_matrix, extrinsic_matrix_inv, image, image_height, image_width):
    # Project Pointcloud on image
    # Project 3D points in vehicle reference to 3D points in camera reference using extrinsic params
    point_cloud_hom = cart2hom(pointcloud)  # nx4
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
    point_cloud_cam_z = point_cloud_cam[point_cloud_cam[:, 2] >= 0]
    pointcloud_label = pointcloud_label[point_cloud_cam[:, 2] >= 0]

    # Project 3D points in camera reference in 2D points in image reference using intrinsic params
    point_cloud_cam_hom = cart2hom(point_cloud_cam_z)
    # Compute perspective projection matrix
    P = np.hstack((intrinsic_matrix, np.zeros((3, 1))))
    point_cloud_image = np.dot(P, point_cloud_cam_hom.T).T
    point_cloud_image = point_cloud_image[:, :2] / point_cloud_image[:, 2:]
    rospy.loginfo("Pointcloud projected from 3D vehicle frame to 3D camera frame")

    # Filtered 2D points > remove points out of the image FOV
    filtered_indices = (
        (point_cloud_image[:, 0] >= 0) & (point_cloud_image[:, 0] < image_width) &
        (point_cloud_image[:, 1] >= 0) & (point_cloud_image[:, 1] < image_height)
    )
    point_cloud_image = point_cloud_image[filtered_indices]
    pointcloud_label = pointcloud_label[filtered_indices]
    rospy.loginfo("Pointcloud projected from 3D camera frame to 2D image frame")

    # Publish results for RVIZ visualization
    detection_association_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    circle_radius = 2
    for point, label in zip(point_cloud_image, pointcloud_label):
        x, y = int(point[0]), int(point[1])
        cv2.circle(detection_association_image, (x, y), circle_radius, (int(label[2]), int(label[1]), int(label[0])), -1)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('result', detection_association_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_referenced_pointcloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame])

if __name__ == "__main__":
    rospy.init_node('data_association', anonymous=True)
    rospy.loginfo("Node initialized correctly")
    
    # Specify the paths to the rosbags
    pointcloud_bag_path     = os.path.join(src_dir, "dataset/pointcloud_bags")
    image_bag_path          = os.path.join(src_dir, "dataset/camera_images_bags")
    camera_proj_bag_path    = os.path.join(src_dir, "dataset/camera_params_bags")

    bag_pointcloud_files    = sorted([os.path.join(pointcloud_bag_path, f) for f in os.listdir(pointcloud_bag_path) if f.endswith('.bag')])
    bag_images_files        = sorted([os.path.join(image_bag_path, f) for f in os.listdir(image_bag_path) if f.endswith('.bag')])
    bag_camera_params_files = sorted([os.path.join(camera_proj_bag_path, f) for f in os.listdir(camera_proj_bag_path) if f.endswith('.bag')])


    for pointcloud_bag_file, images_bag_file, camera_params_bag_file in zip(bag_pointcloud_files, bag_images_files, bag_camera_params_files):
        rospy.loginfo("Reading bag file: %s", pointcloud_bag_file)
        with rosbag.Bag(pointcloud_bag_file, 'r') as pc_bag:
            for topic, msg, t in pc_bag.read_messages():
                pointcloud, cluster_labels = process_pointcloud(msg)
                rospy.loginfo("Received message from topic %s", topic)
        
        # Visualize labeled pointcloud
        # plot_referenced_pointcloud(pointcloud)

        rospy.loginfo("Reading bag file: %s", camera_params_bag_file)
        with rosbag.Bag(camera_params_bag_file, 'r') as cam_bag:
            for topic, msg, t in cam_bag.read_messages():
                intrinsic_matrix, extrinsic_matrix, extrinsic_matrix_inv = process_camera(msg)
                rospy.loginfo("Received message from topic %s", topic)

        rospy.loginfo("Reading bag file: %s", images_bag_file)
        with rosbag.Bag(images_bag_file, 'r') as image_bag:
            for topic, msg, t in image_bag.read_messages():
                image, image_height, image_width = process_image(msg)
                rospy.loginfo("Received message from topic %s", topic)

        # Data association
        associate_data(pointcloud, cluster_labels, intrinsic_matrix, extrinsic_matrix, extrinsic_matrix_inv, image, image_height, image_width)
