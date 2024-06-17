#!/usr/bin/env python3
"""
    This node reads a point cloud from Waymo Open Dataset, converts it to PCL, and sends it to various service servers for processing.
"""
import os
import sys
import numpy as np
import pathlib
import struct
import ctypes
import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image
from waymo_parser.msg import CameraProj
from pointcloud_clustering.srv import clustering_srv, clustering_srvRequest, landmark_detection_srv, landmark_detection_srvRequest
from cv_bridge import CvBridge
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_3d_parser import *

class WaymoClient:
    """
    This class manages the communication with the server processes
    as well as the parsing of the data from the source (DataBase or sensors).
    """
    def __init__(self, frame, cams_order):
        self.frame = frame
        self.cams_order = cams_order
        self.points = np.zeros((0,3)) # Original pointcloud obtained from the dataset
        self.pointcloud = np.zeros((0, 3)) # Pointcloud processor response (clustered pointcloud)
        self.cluster_labels = np.zeros((0, 3)) # Pointcloud processor response (cluster labels assingment)
        self.init_camera_params()
        self.processed_image = None # Landmark detection response (processed image)
        self.image_height = None
        self.image_width = None
        self.detection_association_image = None # Image resultant of projecting the pointcloud in the original image

        rospy.loginfo("Waiting for server processes...")
        # Initialize pointcloud process
        rospy.wait_for_service('process_pointcloud')
        self.pointcloud_clustering_client = rospy.ServiceProxy('process_pointcloud', clustering_srv)
        rospy.loginfo("Clustering service is running")

        # Initialize landmark process
        rospy.wait_for_service('landmark_detection')
        self.landmark_detection_client = rospy.ServiceProxy('landmark_detection', landmark_detection_srv)
        rospy.loginfo("Landmark service is running")
    
    def init_camera_params(self):
        camera_intrinsic = np.array(self.frame.context.camera_calibrations[0].intrinsic)
        self.intrinsic_matrix = [[camera_intrinsic[0], 0, camera_intrinsic[2]],
                            [0, camera_intrinsic[1], camera_intrinsic[3]],
                            [0, 0, 1]]
        distortion = np.array([camera_intrinsic[4], camera_intrinsic[5], camera_intrinsic[6], camera_intrinsic[7], camera_intrinsic[8]])

        self.extrinsic_matrix = np.array((self.frame.context.camera_calibrations[0].extrinsic.transform), dtype=np.float32).reshape(4, 4)
        # Extract rotation and translation components from the extrinsic matrix
        rotation = self.extrinsic_matrix[:3, :3]
        translation = self.extrinsic_matrix[:3, 3]
        # Invert the rotation matrix
        rotation_inv = np.linalg.inv(rotation)
        # Invert the translation
        translation_inv = -np.dot(rotation_inv, translation)
        # Construct the new extrinsic matrix (from camera to vehicle)
        self.extrinsic_matrix_inv = np.zeros((4, 4), dtype=np.float32)
        self.extrinsic_matrix_inv[:3, :3] = rotation_inv
        self.extrinsic_matrix_inv[:3, 3] = translation_inv
        self.extrinsic_matrix_inv[3, 3] = 1.0
        self.extrinsic_matrix_inv = self.extrinsic_matrix_inv[:3,:]

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Output: nx4 points in Homogeneous by appending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def process_pointcloud(self):
        """
        Gets the pointcloud from the source and manages the processing with the server service.
        """
        self.points, points_cp = self.pointcloud_processor.get_pointcloud()
        if self.points is not None:
            self.pointcloud_processor.pointcloud_to_ros(self.points)
            request = clustering_srvRequest()
            request.pointcloud = self.pointcloud_processor.pointcloud_msg

            try:
                rospy.loginfo("Calling pointcloud clustering service...")
                response = self.pointcloud_clustering_client(request)
                clustered_pointcloud = response.clustered_pointcloud
                rospy.loginfo("Pointcloud received")
                self.pointcloud, self.cluster_labels = self.pointcloud_processor.respmsg_to_pointcloud(clustered_pointcloud)
                rospy.loginfo("Pointcloud message decoded")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        else:
            rospy.loginfo("No pointcloud in frame")

    def process_image(self):
        image = self.camera_processor.get_camera_image()[0]
        if image is not None:
            self.camera_processor.camera_to_ros(image)
            request = landmark_detection_srvRequest()
            request.image = self.camera_processor.camera_msg

            try:
                rospy.loginfo("Calling landmark detection service...")
                response = self.landmark_detection_client(request)
                image_detection = response.processed_image
                rospy.loginfo("Detection received")
                self.processed_image = self.camera_processor.respmsg_to_image(image_detection)
                self.image_height, self.image_width, _ = self.processed_image.shape
                rospy.loginfo("Image message decoded")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        else:
            rospy.loginfo("No image in frame")

    def project_pointcloud_on_image(self):
        if self.pointcloud.size == 0 or self.processed_image is None:
            rospy.logwarn("No pointcloud or processed image available for projection")
            return

        # Project Pointcloud on image
        point_cloud_hom = self.cart2hom(self.pointcloud)  # nx4
        point_cloud_cam = np.dot(point_cloud_hom, np.transpose(self.extrinsic_matrix_inv))

        # Rotate point cloud to match pinhole model axis
        theta_x = np.pi / 2
        theta_y = -np.pi / 2
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
        positive_indices = (point_cloud_cam[:, 2] >= 0)
        point_cloud_cam = point_cloud_cam[positive_indices]
        positive_labels = self.cluster_labels[positive_indices]
        # Project 3D points in camera reference in 2D points in image reference using intrinsic params
        point_cloud_cam_hom = self.cart2hom(point_cloud_cam)
        # Compute perspective projection matrix
        P = np.hstack((self.intrinsic_matrix, np.zeros((3, 1))))
        point_cloud_image = np.dot(P, point_cloud_cam_hom.T).T
        point_cloud_image = point_cloud_image[:, :2] / point_cloud_image[:, 2:]
        rospy.loginfo("Pointcloud projected from 3D vehicle frame to 3D camera frame")

        # Filtered 2D points > remove points out of the image FOV
        filtered_indices = (
            (point_cloud_image[:, 0] >= 0) & (point_cloud_image[:, 0] < self.image_width) &
            (point_cloud_image[:, 1] >= 0) & (point_cloud_image[:, 1] < self.image_height)
        )
        point_cloud_image = point_cloud_image[filtered_indices]
        filtered_labels = positive_labels[filtered_indices]
        rospy.loginfo("Pointcloud projected from 3D camera frame to 2D image frame")

        # Publish results for RVIZ visualization
        self.detection_association_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        circle_radius = 2
        for point, color in zip(point_cloud_image, filtered_labels):
            x, y = int(point[0]), int(point[1])
            circle_color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.circle(self.detection_association_image, (x, y), circle_radius, circle_color, -1)


class PointCloudProcessor(WaymoClient):
    def __init__(self, frame):
        super().__init__(frame, None)
        self.pointcloud_msg = self.init_pointcloud_msg()

    def init_pointcloud_msg(self):
        pointcloud_msg = PointCloud2()
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 16
        pointcloud_msg.height = 1
        pointcloud_msg.is_dense = True
        return pointcloud_msg

    def get_pointcloud(self):
        range_images, camera_projections, segmentation_labels, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(self.frame)
        points_return1 = self._range_image_to_pcd(self.frame, range_images, camera_projections, range_image_top_pose)
        points_return2 = self._range_image_to_pcd(self.frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
        points, points_cp = concatenate_pcd_returns(points_return1, points_return2)
        return points, points_cp
    
    def _range_image_to_pcd(self, frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
        points, points_cp = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=ri_index)
        return points, points_cp

    def pointcloud_to_ros(self, points):
        self.pointcloud_msg.width = points.shape[0]
        self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * self.pointcloud_msg.width
        data = np.column_stack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
        self.pointcloud_msg.data = data.tobytes()

    def respmsg_to_pointcloud(self, msg):
        pointcloud = np.zeros((0, 3))
        cluster_labels = np.zeros((0, 3))
        gen = pc2.read_points(msg, skip_nans=True)
        for x in gen:
            test = x[3]
            s = struct.pack('>f', test)
            i = struct.unpack('>l', s)[0]
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)
            pointcloud = np.append(pointcloud, [[x[0], x[1], x[2]]], axis=0)
            cluster_labels = np.append(cluster_labels, [[r, g, b]], axis=0)

        return pointcloud, cluster_labels

class CameraProcessor(WaymoClient):
    def __init__(self, frame):
        super().__init__(frame, None)
        self.camera_msg = self.init_camera_msg()

    def init_camera_msg(self):
        camera_msg = Image()
        camera_msg.encoding = "bgr8"
        camera_msg.is_bigendian = False
        return camera_msg

    def get_camera_image(self):
        cameras_images = {image.name: get_frame_image(image)[..., ::-1] for image in self.frame.images}
        ordered_images = [cameras_images[1]]
        return ordered_images

    def camera_to_ros(self, image):
        self.camera_msg.height = image.shape[0]
        self.camera_msg.width = image.shape[1]
        self.camera_msg.step = image.shape[1] * 3
        self.camera_msg.data = image.tobytes()

    def respmsg_to_image(self, msg):
        bridge = CvBridge()
        processed_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        return processed_image


def plot_referenced_pointcloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame])

if __name__ == "__main__":
    rospy.init_node('waymo_client', anonymous=True)
    rospy.loginfo("Node initialized correctly")

    rate = rospy.Rate(1/30)  # Adjust the rate as needed

    dataset_path = os.path.join(src_dir, "dataset/waymo_valid_scene")
    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))


    while not rospy.is_shutdown():
        for scene_index, scene_path in enumerate(tfrecord_list):
            scene_name = scene_path.stem
            rospy.loginfo(f"Scene {scene_index}: {scene_name} processing: {scene_path}")
            frame = next(load_frame(scene_path))
            if frame is not None:
                wc = WaymoClient(frame, [2, 1, 3])
                wc.pointcloud_processor = PointCloudProcessor(frame)
                wc.camera_processor = CameraProcessor(frame)

                rospy.loginfo("Calling processing services")

                # Pointcloud processing
                rospy.loginfo("Pointcloud processing service")
                wc.pointcloud_processor.pointcloud_msg.header.frame_id = f"base_link_{scene_name}"
                wc.pointcloud_processor.pointcloud_msg.header.stamp = rospy.Time.now()
                wc.process_pointcloud()
                if wc.pointcloud.size < 0:
                    rospy.logerror("Pointcloud received is empty")
                    continue
                rospy.loginfo("Processed Pointcloud received")
                plot_referenced_pointcloud(wc.pointcloud)

                # Camera processing
                rospy.loginfo("Landmark detection processing service")
                wc.camera_processor.camera_msg.header.frame_id = f"base_link_{scene_name}"
                wc.camera_processor.camera_msg.header.stamp = rospy.Time.now()
                wc.process_image()
                if wc.processed_image is None:
                    rospy.logerror("Image received is empty")
                    continue
                cv2.namedWindow('detection result', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('detection result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('detection result', wc.processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Project camera on image
                wc.project_pointcloud_on_image()
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('result', wc.detection_association_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                rate.sleep()
