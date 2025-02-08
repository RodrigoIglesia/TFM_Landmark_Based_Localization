#!/usr/bin/env python3
"""
This is the main service of LandmarkBsedLocalization project
Is in charge of parsing several data from waymo open dataset and send it to different services to process it
- Odometry > reads position information of the vehicle on each frame and applies a gaussian noise to simulate odometry error
- Pointcloud > reads an input pointcloud and sends it pointcloud clustering service to obtain clustered landmards
- Landmark detection > reads de central camera image form WOD and sends it to the image analysis process to obtain semantic segmentations of landmarks
- Data association > Associates pointcloud clustering and image analysis information to get a real representation of the landmarks
- Data fusion > sends to the data fusion process Odometry pose and Landmark poses > receives the corrected pose of the vehicle

Every element is represented in the vehicle frame
"""

# IMPORTANT:    Landmarks are represented in vehicle and map frame, global frame is never used.
import os
import sys
import numpy as np
import math
import pathlib
import struct
import ctypes
import rospy
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from pointcloud_clustering.msg import positionRPY
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import PoseStamped, PoseArray, Point, Quaternion

from pointcloud_clustering.srv import clustering_srv, clustering_srvRequest, landmark_detection_srv, landmark_detection_srvRequest, data_fusion_srv, data_fusion_srvRequest
from cv_bridge import CvBridge
import cv2
import csv
import open3d as o3d
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.publisher_utils import *
import waymo_utils.waymo_3d_parser as w3d
import waymo_utils.transform_utils as tu

import configparser

def load_config():
    config_file = rospy.get_param("~client_config_file_path")
    config = configparser.ConfigParser()
    config.read(config_file)
    try:
        position_noise_std = float(config["NOISE"]["position_noise_std"])
        orientation_noise_std = float(config["NOISE"]["orientation_noise_std"])
    except KeyError as e:
        rospy.logerr(f"Missing parameter in config file: {e}")
        position_noise_std = 0.1  # Deffect value
        orientation_noise_std = 0.005  # Deffect value
    return position_noise_std, orientation_noise_std


class WaymoClient:
    def __init__(self, frame, cams_order):
        self.frame = frame
        self.cams_order = cams_order
        self.points = np.zeros((0,3)) # Received clustered pointcloud
        self.pointcloud = np.zeros((0, 3))
        self.cluster_labels = np.zeros((0, 3))
        self.clustered_pointcloud = {} # Dictionary class:cluster for vehicle frame cluster pointclouds
        self.image = None # Original Image
        self.image_height = None
        self.image_width = None
        self.processed_image = None # Returned segmentation mask by the landmark detection service
        # Positioning
        self.relative_pose = []
        self.odometry_pose = []
        self.corrected_pose = []
        self.position_noise_cumulative = [0, 0, 0]
        self.orientation_noise_cumulative = [0, 0, 0]
        self.relative_cummulative_pose = np.zeros(6)
        self.odometry_cummulative_pose = np.zeros(6)
        self.odometry_path = []
        self.relative_path = []
        self.corrected_path = []
        self.odometry_pose_msg = positionRPY()
        self.landmark_poses_msg_BL = None
        self.previous_transform_matrix = None
        self.transform_matrix = None

        rospy.loginfo("CLIENT Waiting for server processes...")
        # Wait until POINTCLOUD CLUSTERING service is UP AND RUNNING
        rospy.wait_for_service('process_pointcloud')
        # Create client to call POINTCLOUD CLUSTERING
        self.pointcloud_clustering_client = rospy.ServiceProxy('process_pointcloud', clustering_srv)
        rospy.loginfo("CLIENT Clustering service is running")

        # Wait until LANDMARK DETECTION service is UP AND RUNNING
        rospy.wait_for_service('landmark_detection')
        # Create client to call LANDMARK DETECTION
        self.landmark_detection_client = rospy.ServiceProxy('landmark_detection', landmark_detection_srv)
        rospy.loginfo("CLIENT Landmark service is running")

        rospy.wait_for_service('data_fusion')
        self.data_fusion_client = rospy.ServiceProxy('data_fusion', data_fusion_srv)
        rospy.loginfo("CLIENT Data Fusion service is running")

    def process_odometry(self):
        """
        Get incremental pose of the vehicle with constant Gaussian noise.
        This method, based on the global position of the vehicle, computes the relative pose to the previous frame.
        The method adds a Gaussian noise to this increment, so that once the system estimates the pose, it has an error related to the odometry.
        """
        # Extract the transform matrix for the current frame
        self.transform_matrix = np.array(self.frame.pose.transform).reshape(4, 4)

        # Initialize the initial frame as the origin if not already done
        if self.previous_transform_matrix is None:
            rospy.logdebug("Initial frame pose")
            self.relative_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Explicitly set the first pose
            self.odometry_pose = np.array(self.relative_pose)  # Initialize the noisy pose
        else:
            # Compute the relative pose to the previous pose
            relative_transform = np.linalg.inv(self.previous_transform_matrix) @ self.transform_matrix
            self.relative_pose = tu.get_pose(relative_transform)

            # Add constant Gaussian noise to the relative pose
            position_noise = [
                np.random.normal(0, position_noise_std) for _ in range(3)
            ]
            orientation_noise = [
                np.random.normal(0, orientation_noise_std) for _ in range(3)
            ]

            noisy_position = [
                self.relative_pose[i] + position_noise[i] for i in range(3)
            ]
            noisy_orientation = [
                self.relative_pose[i+3] + orientation_noise[i] for i in range(3)
            ]

            self.odometry_pose = noisy_position + noisy_orientation

        # Accumulate the odometry and real pose to get the pose relative to the origin
        self.relative_cummulative_pose = tu.comp_poses(self.relative_cummulative_pose, self.relative_pose)
        self.odometry_cummulative_pose = tu.comp_poses(self.odometry_cummulative_pose, self.odometry_pose)

        rospy.logdebug(f"CLIENT Vehicle relative (real) pose: {self.relative_cummulative_pose}")
        rospy.loginfo(f"CLIENT Vehicle odometry INCREMENTAL pose: {self.odometry_pose}")
        rospy.loginfo(f"CLIENT Vehicle odometry pose: {self.odometry_cummulative_pose}")

        # Add poses to generate a path
        self.relative_path.append(self.relative_cummulative_pose.tolist())
        self.odometry_path.append(self.odometry_cummulative_pose.tolist())
        self.previous_transform_matrix = self.transform_matrix  # Update for the next iteration


    def process_pointcloud(self):
        self.clustered_pointcloud.clear()
        self.points = self.pointcloud_processor.get_pointcloud()
        if self.points is not None:
            self.pointcloud_processor.pointcloud_to_ros(self.points)
            request = clustering_srvRequest()
            request.pointcloud = self.pointcloud_processor.pointcloud_msg

            try:
                rospy.loginfo("CLIENT Calling pointcloud clustering service...")
                response = self.pointcloud_clustering_client(request)
                clustered_pointcloud = response.clustered_pointcloud
                rospy.logdebug("Pointcloud received")
                self.pointcloud, self.cluster_labels = self.pointcloud_processor.respmsg_to_pointcloud(clustered_pointcloud)
                rospy.logdebug("Pointcloud message decoded")

            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")

            # Get the dictionary of lables and clusters
            unique_labels = np.unique(self.cluster_labels, axis=0)
            self.clustered_pointcloud = {tuple(label): self.pointcloud[np.all(self.cluster_labels == label, axis=1), :] for label in unique_labels}

        else:
            rospy.logdebug("No pointcloud in frame")


    def process_image(self):
        self.image = self.camera_processor.get_camera_image()[0]
        if self.image is not None:
            self.camera_processor.camera_to_ros(self.image)
            request = landmark_detection_srvRequest()
            request.image = self.camera_processor.camera_msg

            try:
                rospy.loginfo("CLIENT Calling landmark detection service...")
                response = self.landmark_detection_client(request)
                image_detection = response.processed_image
                rospy.logdebug("Detection received")
                self.processed_image = self.camera_processor.respmsg_to_image(image_detection)
                self.image_height, self.image_width, _ = self.processed_image.shape
                rospy.logdebug("Image message decoded")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        else:
            rospy.logdebug("No image in frame")


    def process_EKF(self, clusters_poses):
        """
        Method to send current pose and rest of parameters to the EKF process
        """
        self.landmark_poses_msg_BL = PoseArray()
        # Create the service request
        ekf_request = data_fusion_srvRequest()
        
        # Populate the request with incremental odometry pose
        self.odometry_pose_msg.x        = self.odometry_pose[0]
        self.odometry_pose_msg.y        = self.odometry_pose[1]
        self.odometry_pose_msg.z        = self.odometry_pose[2]
        self.odometry_pose_msg.roll     = self.odometry_pose[3]
        self.odometry_pose_msg.pitch    = self.odometry_pose[4]
        self.odometry_pose_msg.yaw      = self.odometry_pose[5]
        self.odometry_pose_msg.stamp    = rospy.Time.now()
        ekf_request.odometry = self.odometry_pose_msg
        
        rospy.logdebug(f"Waymo Client Incremental odometry sent: [{self.odometry_pose[0]}, {self.odometry_pose[1]}, {self.odometry_pose[2]}, {self.odometry_pose[3]}, {self.odometry_pose[4]}, {self.odometry_pose[5]}, {rospy.Time.now()}]")
        rospy.logdebug(f"Waymo Client Incremental odometry sent in message: [{self.odometry_pose_msg.x}, {self.odometry_pose_msg.y}, {self.odometry_pose_msg.z}, {self.odometry_pose_msg.roll}, {self.odometry_pose_msg.pitch}, {self.odometry_pose_msg.yaw}, {self.odometry_pose_msg.stamp}]")


        # Populate the request with landmark poses in Base Line frame
        for label, pose in clusters_poses.items():
            landmark_pose_BL = PoseStamped()
            landmark_pose_BL.pose.position.x = pose[0]
            landmark_pose_BL.pose.position.y = pose[1]
            landmark_pose_BL.pose.position.z = pose[2]
            quaternion = R.from_euler('xyz', pose[3:]).as_quat()
            landmark_pose_BL.pose.orientation.x = quaternion[0]
            landmark_pose_BL.pose.orientation.y = quaternion[1]
            landmark_pose_BL.pose.orientation.z = quaternion[2]
            landmark_pose_BL.pose.orientation.w = quaternion[3]
            self.landmark_poses_msg_BL.poses.append(landmark_pose_BL.pose)

        ekf_request.verticalElements_BL = self.landmark_poses_msg_BL

        # Call the EKF service
        try:
            rospy.loginfo("CLIENT Calling EKF service...")
            ekf_response = self.data_fusion_client(ekf_request)
            rospy.logdebug("EKF service call successful")
            
            # Handle the response to get the corrected pose
            position_rpy = ekf_response.corrected_position

            self.corrected_pose = [
                position_rpy.x, 
                position_rpy.y, 
                position_rpy.z, 
                position_rpy.roll, 
                position_rpy.pitch, 
                position_rpy.yaw
            ]
            
            rospy.loginfo(f"CLIENT Corrected EKF Pose: {self.corrected_pose}")

            # Add the corrected pose to a path array
            self.corrected_path.append(self.corrected_pose)

        except rospy.ServiceException as e:
            rospy.logerr(f"EKF service call failed: {e}")


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
        points_return1, _ = self._range_image_to_pcd(self.frame, range_images, camera_projections, range_image_top_pose)
        points_return2, _ = self._range_image_to_pcd(self.frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
        # points = w3d.concatenate_pcd_returns(points_return1, points_return2)
        non_empty_points = [arr for arr in points_return1 if arr.size != 0]
        points_all = np.concatenate(non_empty_points, axis=0)
        return points_all
    
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


class DataAssociationProcessor:
    def __init__(self, frame):
        self.frame = frame
        self.image_height = None
        self.image_width = None
        self.clustered_pointcloud_image = {}
        self.clustered_pointcloud_iou = {}
        self.clustered_pointcloud_iou_vehicle_frame = {}
        self.clusters_poses = {}
        self.init_camera_params()

    def init_camera_params(self):
        camera_intrinsic = np.array(self.frame.context.camera_calibrations[0].intrinsic)
        self.intrinsic_matrix = [[camera_intrinsic[0], 0, camera_intrinsic[2]],
                            [0, camera_intrinsic[1], camera_intrinsic[3]],
                            [0, 0, 1]]
        distortion = np.array([camera_intrinsic[4], camera_intrinsic[5], camera_intrinsic[6], camera_intrinsic[7], camera_intrinsic[8]]) #TODO: Adjust projections using the distortion parameters

        self.extrinsic_matrix = np.array((self.frame.context.camera_calibrations[0].extrinsic.transform), dtype=np.float32).reshape(4, 4)
        rotation = self.extrinsic_matrix[:3, :3]
        translation = self.extrinsic_matrix[:3, 3]
        rotation_inv = np.linalg.inv(rotation)
        translation_inv = -np.dot(rotation_inv, translation)
        self.extrinsic_matrix_inv = np.zeros((4, 4), dtype=np.float32)
        self.extrinsic_matrix_inv[:3, :3] = rotation_inv
        self.extrinsic_matrix_inv[:3, 3] = translation_inv
        self.extrinsic_matrix_inv[3, 3] = 1.0
        self.extrinsic_matrix_inv = self.extrinsic_matrix_inv[:3,:]

    def __vehicle_to_camera(self, pointcloud, transform):
        point_cloud_hom = tu.cart2hom(pointcloud)
        point_cloud_cam = np.dot(point_cloud_hom, np.transpose(transform))

        theta_x = np.pi / 2
        theta_y = -np.pi / 2
        Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        point_cloud_cam = np.dot(Rx, point_cloud_cam.T).T
        point_cloud_cam = np.dot(Ry, point_cloud_cam.T).T

        return point_cloud_cam

    def project_pointcloud_on_image(self, pointcloud, cluster_labels):
        if pointcloud.size == 0:
            rospy.logerr("No pointcloud available for projection")
            return

        point_cloud_cam = self.__vehicle_to_camera(pointcloud, self.extrinsic_matrix_inv)

        positive_indices = (point_cloud_cam[:, 2] >= 0)
        point_cloud_cam = point_cloud_cam[positive_indices]
        positive_labels = cluster_labels[positive_indices]
        point_cloud_cam_hom = tu.cart2hom(point_cloud_cam)
        P = np.hstack((self.intrinsic_matrix, np.zeros((3, 1))))
        point_cloud_image = np.dot(P, point_cloud_cam_hom.T).T
        point_cloud_image = point_cloud_image[:, :2] / point_cloud_image[:, 2:]

        filtered_indices = (
            (point_cloud_image[:, 0] >= 0) & (point_cloud_image[:, 0] < self.image_width) &
            (point_cloud_image[:, 1] >= 0) & (point_cloud_image[:, 1] < self.image_height)
        )
        point_cloud_image = point_cloud_image[filtered_indices]
        filtered_labels = positive_labels[filtered_indices]

        unique_labels = np.unique(filtered_labels, axis=0)
        self.clustered_pointcloud_image = {tuple(label): point_cloud_image[np.all(filtered_labels == label, axis=1), :] for label in unique_labels}

    def __calculate_iou(self, hull1, hull2):
        polygon1 = Polygon(hull1.reshape(-1, 2))
        polygon2 = Polygon(hull2.reshape(-1, 2))
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        return intersection / union if union != 0 else 0

    def __draw_iou_text(self, image, hull1, hull2, iou):
        center1 = np.mean(hull1.reshape(-1, 2), axis=0).astype(int)
        center2 = np.mean(hull2.reshape(-1, 2), axis=0).astype(int)
        midpoint = ((center1 + center2) // 2).astype(int)
        cv2.putText(image, f'IoU: {iou:.2f}', tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def filter_association_iou(self, processed_image, clustered_pointcloud, iou_threshold=0.2, min_hull_area=500, debug=False):
        self.clustered_pointcloud_iou.clear()
        self.clustered_pointcloud_iou_vehicle_frame.clear()

        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation_hulls = [cv2.convexHull(contour).reshape((-1, 2)) for contour in contours]

        for label, cluster_points in self.clustered_pointcloud_image.items():
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                cluster_hull = cluster_points[hull.vertices].reshape((-1, 2)).astype(np.int32)
                if cv2.contourArea(cluster_hull) <= min_hull_area:
                    continue

                for seg_hull in segmentation_hulls:
                    if cv2.contourArea(seg_hull) <= min_hull_area:
                        continue
                    iou = self.__calculate_iou(cluster_hull, seg_hull)
                    if iou > iou_threshold:
                        self.clustered_pointcloud_iou[label] = cluster_points
                        if debug:
                            pair_image = processed_image.copy()
                            cv2.drawContours(pair_image, [seg_hull], -1, (255, 0, 0), 2)
                            cv2.drawContours(pair_image, [cluster_hull], -1, (0, 0, 255), 2)
                            self.__draw_iou_text(pair_image, cluster_hull, seg_hull, iou)
                        self.clustered_pointcloud_iou_vehicle_frame[label] = clustered_pointcloud[label]

    def calculate_landmark_pose(self):
        self.clusters_poses = {}
        for label, pointcloud in self.clustered_pointcloud_iou_vehicle_frame.items():
            centroid = w3d.get_cluster_centroid(pointcloud)
            roll, pitch, yaw = 0.0, 0.0, 0.0
            self.clusters_poses[label] = [centroid[0], centroid[1], centroid[2], roll, pitch, yaw]


if __name__ == "__main__":
    ##############################################################################################
    ## Load configuration
    ##############################################################################################
    position_noise_std, orientation_noise_std = load_config()

    ##############################################################################################
    ## Initialize main client node
    ##############################################################################################
    rospy.init_node('waymo_client', anonymous=True)
    rospy.loginfo("CLIENT Node initialized correctly")

    rate = rospy.Rate(1/4)

    #TODO: REMOVE > Load map pointcloud
    try:
        map_pc_file_path = rospy.get_param('map_pointcloud')
    except KeyError as e:
        rospy.logerr(f"No param: {e}")
    except rospy.ROSException as e:
        rospy.logerr(f"Error reading param: {e}")
    map_points = []
    map_published = False
    with open(map_pc_file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 3:
                x, y, z = map(float, row[:3])
                map_points.append((x, y, z))

    # Read dataset
    try:
        scene_path = rospy.get_param('scene')
    except KeyError as e:
        rospy.logerr(f"No param: {e}")
    except rospy.ROSException as e:
        rospy.logerr(f"Error reading param: {e}")

    # Initialize classes outside the loop
    wc = None
    pointcloud_processor = None
    camera_processor = None
    data_association_processor = None

    ##############################################################################################
    ## Dataset Parsing
    ##############################################################################################
    while not rospy.is_shutdown():
        scene_name = pathlib.Path(scene_path).stem
        rospy.loginfo(f"CLIENT Scene: {scene_name} processing: {scene_path}")
        frame_n = 0  # Scene frames counter

        # Create a folder to save the results with today datetime
        today_date = datetime.now().strftime("%d%m%Y%H%M")

        # Crear el nombre de la carpeta
        results_folder = os.path.join(src_dir, f"results/{scene_name}/{today_date}")

        # Crear la carpeta si no existe
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        else:
            rospy.logwarn(f"Folder'{results_folder}' already exists.")

        # Initialize CSV file and write header at the beginning of each scene
        csv_file_path = os.path.join(results_folder, f'poses_{scene_name}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header once at the beginning
            header = [
                'frame', 'real_x', 'real_y', 'real_z', 'real_roll', 'real_pitch', 'real_yaw',
                'odometry_x', 'odometry_y', 'odometry_z', 'odometry_roll', 'odometry_pitch', 'odometry_yaw',
                'corrected_x', 'corrected_y', 'corrected_z', 'corrected_roll', 'corrected_pitch', 'corrected_yaw'
            ]
            csv_writer.writerow(header)
        #TODO: REMOVE > Save detected landmarks
        landmark_csv_file_path = os.path.join(results_folder, f'landmarks_{scene_name}.csv')
        with open(landmark_csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header once at the beginning
            header = [
                'frame', 'Landmark_X', 'Landmark_Y', 'Landmark_Z', 'Landmark_Roll', 'Landmark_Pitch', 'Landmark_Yaw'
            ]
            csv_writer.writerow(header)

        # Reset odometry cumulative error every scene
        #TODO Odometría simulada, en caso de tener odometría con error real > eliminar
        if wc is not None:
            wc.position_noise_cumulative = [0, 0, 0]
            wc.orientation_noise_cumulative = [0, 0, 0]

        for frame in load_frame(scene_path):
            if frame is None:
                rospy.logerr("Error reading frame")
                continue

            # Initialize classes if not already done
            if wc is None:
                wc = WaymoClient(frame, [2, 1, 3])
            if pointcloud_processor is None:
                pointcloud_processor = PointCloudProcessor(frame)
            if camera_processor is None:
                camera_processor = CameraProcessor(frame)
            if data_association_processor is None:
                data_association_processor = DataAssociationProcessor(frame)


            # Update frame for each class
            wc.frame = frame
            pointcloud_processor.frame = frame
            camera_processor.frame = frame
            data_association_processor.frame = frame

            # Set class references in wc
            wc.pointcloud_processor = pointcloud_processor
            wc.camera_processor = camera_processor

            ##############################################################################################
            ## Procesing Services
            ##############################################################################################

            rospy.logdebug("Calling processing services")
            frame_n += 1
            rospy.loginfo(f"\n CLIENT New Frame {frame_n} \n")

            #TODO: Debug - Publish map
            ##################################################
            #TODO: Remove Plot saved map elements
            header = Header()
            header.frame_id = "map"
            header.stamp = rospy.Time.now()
            publish_pointcloud_to_topic('map_pointcloud', map_points, header)

            """ 
            PROCESS ODOMETRY --> GET INCREMENTAL VEHICLE POSITION ON EACH FRAME
            Odometry service also get the transformation matrix of the vehicle
            In this version, incremental odometry is also calculated by the client service
            """

            # Obtain odometry increment (in frame 0 will be 0)
            wc.process_odometry(position_noise_std, orientation_noise_std)
            #TODO: REMOVE
            relative_pose = np.array(wc.relative_cummulative_pose)

            # DEBUG -> send incremental odometry to publish in RVIZ
            text = f"Frame: ({frame_n})\n" \
                                f"Pos: ({relative_pose[0]:.2f}, {relative_pose[1]:.2f}, {relative_pose[2]:.2f})\n" \
                                f"Orientation: ({relative_pose[3]:.2f}, {relative_pose[4]:.2f}, {relative_pose[5]:.2f})"
            header = Header()
            header.frame_id = "base_link"
            header.stamp = rospy.Time.now()
            publish_incremental_pose_to_topic("vehicle_pose", relative_pose, text, header)
            odometry_pose = np.array(wc.odometry_cummulative_pose)

            # DEBUG -> send incremental odometry to publish in RVIZ
            text = f"Frame: ({frame_n})\n" \
                                f"Pos: ({odometry_pose[0]:.2f}, {odometry_pose[1]:.2f}, {odometry_pose[2]:.2f})\n" \
                                f"Orientation: ({odometry_pose[3]:.2f}, {odometry_pose[4]:.2f}, {odometry_pose[5]:.2f})"
            header = Header()
            header.frame_id = "base_link"
            header.stamp = rospy.Time.now()
            publish_incremental_pose_to_topic("vehicle_odometry_pose", odometry_pose, text, header)
            
            # DEBUG -> Publish vehicle path
            vehicle_path = np.array(wc.odometry_path)
            publish_path("vehicle_path", vehicle_path, header)

            vehicle_real_path = np.array(wc.relative_path)
            publish_path("vehicle_real_path", vehicle_real_path, header)

            """
            POINTCLOUD CLUSTERING PROCESS
            """
            rospy.logdebug("Pointcloud processing service")
            wc.pointcloud_processor.pointcloud_msg.header.frame_id = f"base_link_{scene_name}"
            wc.pointcloud_processor.pointcloud_msg.header.stamp = rospy.Time.now()
            wc.process_pointcloud()
            if wc.pointcloud.size < 0:
                rospy.logerr("Pointcloud received is empty")
                continue
            rospy.logdebug("Processed Pointcloud received")

            """
            LANDMARK DETECTION PROCESS
            """
            rospy.logdebug("Landmark detection processing service")
            wc.camera_processor.camera_msg.header.frame_id = f"base_link_{scene_name}"
            wc.camera_processor.camera_msg.header.stamp = rospy.Time.now()
            wc.process_image()
            if wc.processed_image is None:
                rospy.logerr("Image received is empty")
                continue

            """
            DATA ASSOCIATION > PROJECT POINTCLOUD ON IMAGE AND FILTER CLUSTERS
            This process obtains the poses of the detected landmarks referred to the vehicle's pose.
            In this version, this process is carried out by the client service.
            """
            rospy.logdebug("Data association process")
            # Pointcloud - Image projection
            data_association_processor.image_height = wc.image_height
            data_association_processor.image_width = wc.image_width
            data_association_processor.project_pointcloud_on_image(wc.pointcloud, wc.cluster_labels)
            
            # Pointclouds filtering > get the IoU of pointcloud clusters and segmentation masks
            # This method generates a dictionary of the classes and those clusters that match the segmentations.
            # Generates clusters in the camera and vehicle frame
            data_association_processor.filter_association_iou(wc.processed_image, wc.clustered_pointcloud, iou_threshold=0.1)

            # DEBUG - Publish filtered pointcloud only with landmarks
            header = Header()
            header.frame_id = "base_link"
            header.stamp = rospy.Time.now()
            publish_labeled_pointcloud_to_topic('filtered_pointcloud', data_association_processor.clustered_pointcloud_iou_vehicle_frame, header)
            
            # Get cluster landmarks pose
            data_association_processor.calculate_landmark_pose()

            """
            DATA FUSION > CORRECT ODOMETRY POSITION WITH LANDMARK OBSERVATIONS
            """
            rospy.logdebug("Data Fusion processing service")
            wc.process_EKF(data_association_processor.clusters_poses)

            if wc.corrected_pose is None:
                rospy.logerr("No EKF correction received")
                continue

            corrected_pose = wc.corrected_pose
            # wc.previous_transform_matrix = tu.create_homogeneous_matrix(corrected_pose) # Update the previous transform matrix

            # DEBUG - Publish corrected EKF pose and path
            text = f"Frame: ({frame_n})\n" \
                                f"Corrected Pos: ({corrected_pose[0]:.2f}, {corrected_pose[1]:.2f}, {corrected_pose[2]:.2f})\n" \
                                f"Corrected Orientation: ({corrected_pose[3]:.2f}, {corrected_pose[4]:.2f}, {corrected_pose[5]:.2f})"
            header = Header()
            header.frame_id = "base_link"
            header.stamp = rospy.Time.now()
            publish_incremental_pose_to_topic("corrected_vehicle_pose", corrected_pose, text, header)
            
            # DEBUG -> Publish vehicle path
            vehicle_corrected_path = np.array(wc.corrected_path)
            publish_path("corrected_vehicle_path", vehicle_corrected_path, header)

            # Append pose (Referenced to the map Frame) data for the current frame to CSV
            rel_pose = wc.relative_cummulative_pose
            odo_pose = wc.odometry_cummulative_pose
            with open(csv_file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                row = [frame_n], rel_pose[0], rel_pose[1], rel_pose[2], rel_pose[3], rel_pose[4], rel_pose[5],\
                                    odo_pose[0], odo_pose[1], odo_pose[2], odo_pose[3], odo_pose[4], odo_pose[5],\
                                    corrected_pose[0], corrected_pose[1], corrected_pose[2], corrected_pose[3], corrected_pose[4], corrected_pose[5]
                csv_writer.writerow(row)
                rospy.logdebug(f"Frame {frame_n} data appended to CSV")
            
            # Append observations landmarks (Referenced to the map Frame) data for the current frame to CSV
            with open(landmark_csv_file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for label, landmark in data_association_processor.clusters_poses.items():
                    #TODO: Project landmark to map frame
                    landmark = tu.comp_poses(corrected_pose, landmark)
                    row = [frame_n], landmark[0], landmark[1], landmark[2], landmark[3], landmark[4], landmark[5]
                    csv_writer.writerow(row)
                rospy.logdebug(f"Frame {frame_n} data appended to CSV")


            rate.sleep()
        rospy.signal_shutdown("Finished processing all frames")