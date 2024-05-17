#!/usr/bin/env python3
"""
    This node reads a point cloud from Waymo Open Dataset, converts it to PCL, and sends it to various service servers for processing.
"""
import os
import sys

import logging
import numpy as np
import pathlib

import rospy
import rosbag

from sensor_msgs.msg import PointCloud2, PointField, Image
from waymo_parser.msg import CameraProj
from pointcloud_clustering.srv import clustering_srv, clustering_srvRequest, clustering_srvResponse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()


# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *

class WaymoClient:
    def __init__(self, cams_order):
        rospy.loginfo("Initializing client parameters")
        # Vector of integers determining the index order of the cameras, this vector also selects the number of cameras to use
        self.cams_order = cams_order

        # Service clients
        rospy.wait_for_service('process_pointcloud')
        self.pointcloud_clustering_client = rospy.ServiceProxy('process_pointcloud', clustering_srv)
        rospy.loginfo("Clustering service is running")

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
        # Camera projection message
        self.camera_params_msg = CameraProj()
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

        points, points_cp = concatenate_pcd_returns(points_return1, points_return2)

        return points, points_cp
    
    def pointcloud_to_ros(self, points):
        self.pointcloud_msg.width = points.shape[0]
        self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * self.pointcloud_msg.width
        
        # Concatenate points and intensities if 'intensity' is provided
        data = np.column_stack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
        self.pointcloud_msg.data = data.tobytes()

    def process_pointcloud(self, frame):
        points, points_cp = self.get_pointcloud(frame)
        if points is not None:
            # Convert concatenated point cloud to ROS message
            self.pointcloud_to_ros(points)

            # Create a request object
            request = clustering_srvRequest()
            request.pointcloud = self.pointcloud_msg


            # Call the clustering service
            rospy.loginfo("Calling pointcloud clustering service...")
            response = self.pointcloud_clustering_client(request)
            clustered_pointcloud = response.clustered_pointcloud

            return clustered_pointcloud
        else:
            rospy.loginfo("No pointcloud in frame")
            return None

    def get_camera_image(self, frame):
        # Gets and orders camera images from left to right
        cameras_images = {}
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)
            cameras_images[image.name] = decoded_image[...,::-1]
        ordered_images = [cameras_images[1]]

        return ordered_images

    def process_camera(self, frame):
        camera_images = self.get_camera_image(frame)
        rospy.loginfo("Images loaded from dataset")
        # Check whether 1 or more images have been selected
        if len(camera_images) == 1:
            image = camera_images[0]
            self.camera_msg.height = image.shape[0]  # Set image height
            self.camera_msg.width = image.shape[1]  # Set image width
            self.camera_msg.step = image.shape[1] * 3
            self.camera_msg.data = image.tobytes()

            return self.camera_msg
        elif len(camera_images) > 1:
            # TODO: Implement image stitching or process one by one.
            pass
        else:
            rospy.loginfo("No image in frame")
            return None

    def get_camera_params(self, frame):
        camera_intrinsic = np.array(frame.context.camera_calibrations[0].intrinsic)
        camera_extrinsic = np.array(frame.context.camera_calibrations[0].extrinsic.transform)
        self.camera_params_msg.intrinsic = camera_intrinsic
        self.camera_params_msg.extrinsic = camera_extrinsic

        return self.camera_params_msg

if __name__ == "__main__":
    rospy.init_node('waymo_client', anonymous=True)
    rospy.loginfo("Node initialized correctly")

    rate = rospy.Rate(1/30)  # Adjust the rate as needed

    # TODO: Set dataset path as a configurable parameter
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")
    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    wp = WaymoClient([2, 1, 3])

    ##############################################################
    ## Iterate through Dataset Scenes
    ##############################################################
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        scene_name = scene_path.stem
        rospy.loginfo("Scene {}: {} processing: {}".format(str(scene_index), scene_name, scene_path))

        frame = next(load_frame(scene_path))
        if frame is not None:
            rospy.loginfo("Calling processing services")
            # Process point cloud
            wp.pointcloud_msg.header.frame_id = f"base_link_{scene_name}"
            wp.pointcloud_msg.header.stamp = rospy.Time.now()
            clustered_pointcloud = wp.process_pointcloud(frame)
            rospy.loginfo("Processed Pointcloud received")


            # # Process camera parameters
            # wp.camera_params_msg.header.frame_id = f"base_link_{scene_name}"
            # wp.camera_params_msg.header.stamp = rospy.Time.now()
            # camera_params = wp.get_camera_params(frame)


            # # Process camera image
            # wp.camera_msg.header.frame_id = f"base_link_{scene_name}"
            # wp.camera_msg.header.stamp = rospy.Time.now()
            # camera_image = wp.process_camera(frame)
  

            rate.sleep()
