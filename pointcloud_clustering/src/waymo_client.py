#!/usr/bin/env python3
"""
    This node reads a point cloud from Waymo Open Dataset, converts it to PCL, and sends it to various service servers for processing.
"""
# TODO: Incremental position computed is not being plotted correctly in RVIZ, it plots a straight line
import os
import sys
import numpy as np
import math
import pathlib
import struct
import ctypes
import rospy
import open3d as o3d
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import PoseStamped, PoseArray, Point, Quaternion
from nav_msgs.msg import Path, Odometry
from waymo_parser.msg import CameraProj
from pointcloud_clustering.srv import clustering_srv, clustering_srvRequest, landmark_detection_srv, landmark_detection_srvRequest
from cv_bridge import CvBridge
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *
import waymo_utils.waymo_3d_parser as w3d



class WaymoClient:
    def __init__(self, frame, cams_order):
        self.frame = frame
        self.cams_order = cams_order
        self.points = np.zeros((0,3)) # Received clustered pointcloud
        self.pointcloud = np.zeros((0, 3))
        self.cluster_labels = np.zeros((0, 3))
        self.clustered_pointcloud = {} # Dictionary class:cluster for vehicle frame cluster pointclouds
        self.clustered_pointcloud_image = {} # Dictionary class:cluster for image projected pointcloud
        self.clustered_pointcloud_iou = {} # Dictionary class:cluster for image projected pointclouds matched with segmentation masks
        self.clustered_pointcloud_iou_vehicle_frame = {} # Dictionary class:cluster for vehicle frame cluster pointclouds matched with segmentation masks
        self.clusters_poses = {} # Dictionary label:pose of landmarks in vehicle frame
        self.clusters_poses_global = {} # Dictionary label:pose of landmarks in global frame
        self.init_camera_params()
        self.image = None # Original Image
        self.processed_image = None # Returned segmentation mask by the landmark detection service
        self.image_height = None
        self.image_width = None

        rospy.loginfo("Waiting for server processes...")
        rospy.wait_for_service('process_pointcloud')
        self.pointcloud_clustering_client = rospy.ServiceProxy('process_pointcloud', clustering_srv)
        rospy.loginfo("Clustering service is running")

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
        rotation = self.extrinsic_matrix[:3, :3]
        translation = self.extrinsic_matrix[:3, 3]
        rotation_inv = np.linalg.inv(rotation)
        translation_inv = -np.dot(rotation_inv, translation)
        self.extrinsic_matrix_inv = np.zeros((4, 4), dtype=np.float32)
        self.extrinsic_matrix_inv[:3, :3] = rotation_inv
        self.extrinsic_matrix_inv[:3, 3] = translation_inv
        self.extrinsic_matrix_inv[3, 3] = 1.0
        self.extrinsic_matrix_inv = self.extrinsic_matrix_inv[:3,:]

    def cart2hom(self, pts_3d):
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def process_pointcloud(self):
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

            # Get the dictionary of lables and clusters
            unique_labels = np.unique(self.cluster_labels, axis=0)
            self.clustered_pointcloud = {tuple(label): self.pointcloud[np.all(self.cluster_labels == label, axis=1), :] for label in unique_labels}

        else:
            rospy.loginfo("No pointcloud in frame")

    def process_image(self):
        self.image = self.camera_processor.get_camera_image()[0]
        if self.image is not None:
            self.camera_processor.camera_to_ros(self.image)
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

    def __vehicle_to_camera(self, pointcloud, transform):
        point_cloud_hom = self.cart2hom(pointcloud)
        point_cloud_cam = np.dot(point_cloud_hom, np.transpose(transform))

        theta_x = np.pi / 2
        theta_y = -np.pi / 2
        theta_z = 0
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta_x), -np.sin(theta_x)],
                       [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                       [0, 1, 0],
                       [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                       [np.sin(theta_z), np.cos(theta_z), 0],
                       [0, 0, 1]])
        point_cloud_cam = np.dot(Rx, point_cloud_cam.T).T
        point_cloud_cam = np.dot(Ry, point_cloud_cam.T).T

        return point_cloud_cam

    def project_pointcloud_on_image(self):
        """
        Returns: cluster labels dictionary, projected on camera frame --> {label: cluster}
        """
        if self.pointcloud.size == 0 or self.processed_image is None:
            rospy.logwarn("No pointcloud or processed image available for projection")
            return

        point_cloud_cam = self.__vehicle_to_camera(self.pointcloud, self.extrinsic_matrix_inv)

        positive_indices = (point_cloud_cam[:, 2] >= 0)
        point_cloud_cam = point_cloud_cam[positive_indices]
        positive_labels = self.cluster_labels[positive_indices]
        point_cloud_cam_hom = self.cart2hom(point_cloud_cam)
        P = np.hstack((self.intrinsic_matrix, np.zeros((3, 1))))
        point_cloud_image = np.dot(P, point_cloud_cam_hom.T).T
        point_cloud_image = point_cloud_image[:, :2] / point_cloud_image[:, 2:]
        rospy.loginfo("Pointcloud projected from 3D vehicle frame to 3D camera frame")

        filtered_indices = (
            (point_cloud_image[:, 0] >= 0) & (point_cloud_image[:, 0] < self.image_width) &
            (point_cloud_image[:, 1] >= 0) & (point_cloud_image[:, 1] < self.image_height)
        )
        point_cloud_image = point_cloud_image[filtered_indices]
        filtered_labels = positive_labels[filtered_indices]
        rospy.loginfo("Pointcloud projected from 3D camera frame to 2D image frame")

        # Apply convex hull for each cluster
        # Generate clusters based on filtered labels
        unique_labels = np.unique(filtered_labels, axis=0)
        self.clustered_pointcloud_image = {tuple(label): point_cloud_image[np.all(filtered_labels == label, axis=1), :] for label in unique_labels}


    def __calculate_iou(self, hull1, hull2):
        # Convert hulls to polygons, ensure hulls are in the form (N, 2)
        polygon1 = Polygon(hull1.reshape(-1, 2))
        polygon2 = Polygon(hull2.reshape(-1, 2))

        # Calculate intersection and union
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area

        # Calculate IoU
        if union == 0:
            return 0
        return intersection / union


    def __draw_iou_text(self, image, hull1, hull2, iou):
        # Calculate the center points of the hulls
        center1 = np.mean(hull1.reshape(-1, 2), axis=0).astype(int)
        center2 = np.mean(hull2.reshape(-1, 2), axis=0).astype(int)
        
        # Calculate the midpoint between the two centers
        midpoint = ((center1 + center2) // 2).astype(int)
        
        # Draw the IoU value at the midpoint
        cv2.putText(image, f'IoU: {iou:.2f}', tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def filter_association_iou(self, debug=False, min_hull_area = 500, iou_threshold=0.1):
        """
        This method computes the IoU between the clusters projections and the segmentation masks
        Filters the projections and stores only cluster projections that intersect with segmentation masks
        """
        rospy.loginfo("Getting associations...")

        # Convert processed image to grayscale
        processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
        segmentation_contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute convex hulls for the segmentation contours
        segmentation_hulls = [cv2.convexHull(contour).reshape((-1, 2)) for contour in segmentation_contours]

        # Process each cluster
        for label, cluster_points in self.clustered_pointcloud_image.items():
            if len(cluster_points) >= 3:  # Convex hull requires at least 3 points
                hull = ConvexHull(cluster_points)
                cluster_hull = cluster_points[hull.vertices].reshape((-1, 2)).astype(np.int32)

                # Filter cluster hull by area
                if (cv2.contourArea(cluster_hull) <= min_hull_area):
                    continue

                # Calculate IoU with all segmentation hulls
                for seg_hull in segmentation_hulls:
                    # Filter segmentation hulls by area
                    if (cv2.contourArea(seg_hull) <= min_hull_area):
                        continue
                    
                    # Compute IoU
                    iou = self.__calculate_iou(cluster_hull, seg_hull)
                    # Save only clusters which iou with segmentation masks is greater than threshols
                    if (iou > iou_threshold):
                        self.clustered_pointcloud_iou[label] = cluster_points
                    # Filter 3D clusters in vehicle frame
                    self.clustered_pointcloud_iou_vehicle_frame = filtered_dict = {label: self.clustered_pointcloud[label] for label in self.clustered_pointcloud_iou.keys() if label in self.clustered_pointcloud}

                    # Show the image with the pair of hulls
                    if (debug==True):
                        ## TODO: REMOVE DEBUG
                        pair_image = np.copy(self.image)
                        cv2.drawContours(pair_image, [seg_hull.reshape((-1, 1, 2))], -1, (255, 0, 0), 2)  # White for segmentation hull
                        cv2.drawContours(pair_image, [cluster_hull.reshape((-1, 1, 2))], -1, (0, 0, 255), 2)  # Red for cluster hull
                        self.__draw_iou_text(pair_image, cluster_hull, seg_hull, iou)
                        cv2.imshow("Hull Pair", pair_image)
                        cv2.waitKey(0)


    def calculate_landmark_pose(self):
        """
        This method computes the [x,y,z, roll, pitch, yaw] coordinates of a clustered pointcloud
        """
        for label, pointcloud in self.clustered_pointcloud_iou_vehicle_frame.items():
            centroid = w3d.get_cluster_centroid(pointcloud)
            orientation = w3d.get_cluster_orientation(pointcloud)
            
            # Generate dict label:pose
            landmark_pose = [centroid[0], centroid[1], centroid[2], orientation[0], orientation[1], orientation[2]]
            self.clusters_poses[label] = landmark_pose

            # Get pose in global frame
            # Convert the centroid to homogeneous coordinates
            landmark_pose_hom = np.hstack((landmark_pose[:3], [1]))
            landmark_global_hom = np.dot(self.incremental_odometry_extractor.transform_matrix, landmark_pose_hom)
            landmark_global = landmark_global_hom[:3]

            # Rotation
            R_x = np.array([[1, 0, 0],
                    [0, np.cos(orientation[0]), -np.sin(orientation[0])],
                    [0, np.sin(orientation[0]), np.cos(orientation[0])]])

            R_y = np.array([[np.cos(orientation[1]), 0, np.sin(orientation[1])],
                            [0, 1, 0],
                            [-np.sin(orientation[1]), 0, np.cos(orientation[1])]])
            
            R_z = np.array([[np.cos(orientation[2]), -np.sin(orientation[2]), 0],
                            [np.sin(orientation[2]), np.cos(orientation[2]), 0],
                            [0, 0, 1]])
            
            landmark_rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
            # Transform the orientation using the rotation matrix
            global_rotation_matrix = np.dot(self.incremental_odometry_extractor.transform_matrix[:3,:3], landmark_rotation_matrix)

            # Convert the global rotation matrix back to euler angles
            sy = math.sqrt(global_rotation_matrix[0, 0] ** 2 + global_rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                global_roll = math.atan2(global_rotation_matrix[2, 1], global_rotation_matrix[2, 2])
                global_pitch = math.atan2(-global_rotation_matrix[2, 0], sy)
                global_yaw = math.atan2(global_rotation_matrix[1, 0], global_rotation_matrix[0, 0])
            else:
                global_roll = math.atan2(-global_rotation_matrix[1, 2], global_rotation_matrix[1, 1])
                global_pitch = math.atan2(-global_rotation_matrix[2, 0], sy)
                global_yaw = 0

            self.clusters_poses_global[label] = [landmark_global[0],landmark_global[1],landmark_global[2], global_roll, global_pitch, global_yaw]


    def process_EKF(self):
        """
        Method to send current pose and rest of parameters to the EKF process
        """
        pass


class IncrementalOdometryExtractor(WaymoClient):
    def __init__(self, frame):
        super().__init__(frame, None)
        self.transform_matrix = []
        self.prev_transform_matrix = None
        self.global_pose = None
        self.inc_odom = []
        self.inc_odom_msg = PoseStamped()

    def ominus(self, T1, T2):
        return np.linalg.inv(T2) @ T1

    def get_pose(self, T):
        position = T[:3, 3]
        R_matrix = T[:3, :3]
        rotation = R.from_matrix(R_matrix)
        quaternion = rotation.as_quat()

        return [position[0], position[1], position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]

    def process_odometry(self):
        """
        Get incremental pose of the vehicle
        """
        print("Pose transform: ", self.frame.pose.transform)
        self.transform_matrix = np.array(self.frame.pose.transform).reshape(4, 4)
        print("Transform matrix: ", self.transform_matrix)
        self.global_pose = self.get_pose(self.transform_matrix)
        print("Vehicle pose: ", self.get_pose(self.transform_matrix))
        if self.prev_transform_matrix is None:
            # First incremental pose is 0 to send to EKF
            self.inc_odom = [0,0,0,1,0,0,0] # identity quaternion
        else:
            inc_odom_matrix = self.ominus(self.transform_matrix, self.prev_transform_matrix)

            self.inc_odom = self.get_pose(inc_odom_matrix)
            print("Vehicle pose increment: ", self.inc_odom)

        # Create PoseStamped message
        self.inc_odom_msg.pose.position.x       = self.inc_odom[0]
        self.inc_odom_msg.pose.position.y       = self.inc_odom[1]
        self.inc_odom_msg.pose.position.z       = self.inc_odom[2]
        self.inc_odom_msg.pose.orientation.x    = self.inc_odom[3]
        self.inc_odom_msg.pose.orientation.y    = self.inc_odom[4]
        self.inc_odom_msg.pose.orientation.z    = self.inc_odom[5]
        self.inc_odom_msg.pose.orientation.w    = self.inc_odom[6]
        
        self.prev_transform_matrix = self.transform_matrix


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
        points, points_cp = w3d.concatenate_pcd_returns(points_return1, points_return2)
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


if __name__ == "__main__":
    # DEBUG --> Publishers for rviz
    clustered_pointcloud_pub = rospy.Publisher('clustered_pointcloud', PointCloud2, queue_size=10)
    filtered_pointcloud_pub = rospy.Publisher('filtered_pointcloud', PointCloud2, queue_size=10)
    vehicle_pose_pub = rospy.Publisher('vehicle_pose', Odometry, queue_size=10)
    landmark_pose_pub = rospy.Publisher('landmark_pose', PoseArray, queue_size=10)
    landmark_pose_global_pub = rospy.Publisher('landmark_pose_global', PoseArray, queue_size=10)
    pose_text_pub = rospy.Publisher('pose_text', Marker, queue_size=10)

    rospy.init_node('waymo_client', anonymous=True)
    rospy.loginfo("Node initialized correctly")

    rate = rospy.Rate(1/4)

    # Read dataset
    dataset_path = os.path.join(src_dir, "dataset/clustering_test_scene")
    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    # Initialize classes outside the loop
    wc = None
    pointcloud_processor = None
    camera_processor = None
    incremental_odometry_extractor = None

    while not rospy.is_shutdown():
        for scene_index, scene_path in enumerate(tfrecord_list):
            scene_name = scene_path.stem
            rospy.loginfo(f"Scene {scene_index}: {scene_name} processing: {scene_path}")
            frame_n = 0  # Scene frames counter

            ## Evaluation > position and orientation vectors to store vehicles poses
            initial_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64) # DEBUG --> Plot accumulated pose
            accumulated_pose = np.copy(initial_pose)
            for frame in load_frame(scene_path):
                if frame is None:
                    rospy.logerr("Error reading frame")
                    continue
                print("FRAME: ", frame_n)

                # Initialize classes if not already done
                if wc is None:
                    wc = WaymoClient(frame, [2, 1, 3])
                if pointcloud_processor is None:
                    pointcloud_processor = PointCloudProcessor(frame)
                if camera_processor is None:
                    camera_processor = CameraProcessor(frame)
                if incremental_odometry_extractor is None:
                    incremental_odometry_extractor = IncrementalOdometryExtractor(frame)

                # Update frame for each class
                wc.frame = frame
                pointcloud_processor.frame = frame
                camera_processor.frame = frame
                incremental_odometry_extractor.frame = frame

                # Set class references in wc
                wc.pointcloud_processor = pointcloud_processor
                wc.camera_processor = camera_processor
                wc.incremental_odometry_extractor = incremental_odometry_extractor

                rospy.loginfo("Calling processing services")

                """ 
                PROCESS ODOMETRY --> GET INCREMENTAL VEHICLE POSITION ON EACH FRAME
                Odometry service also get the transformation matrix of the vehicle
                """
                wc.incremental_odometry_extractor.process_odometry()

                # DEBUG -> send incremental odometry to publish in RVIZ
                incremental_pose_msg = Odometry()
                # Obtain odometry increment (in frame 0 will be 0)
                increment_pose = np.array(wc.incremental_odometry_extractor.inc_odom)
                  # Update accumulated pose
                accumulated_pose[:3] += increment_pose[:3]
                accumulated_rotation = R.from_quat(accumulated_pose[3:])
                increment_rotation = R.from_quat(increment_pose[3:])
                new_rotation = accumulated_rotation * increment_rotation
                accumulated_pose[3:] = new_rotation.as_quat()
                
                print("NEW VEHICLE POSE RELATIVE TO 0: ", accumulated_pose)

                # Publish the updated pose
                incremental_pose_msg.header.frame_id = "base_link"
                incremental_pose_msg.header.stamp = rospy.Time.now()
                incremental_pose_msg.pose.pose.position = Point(*accumulated_pose[:3])
                quaternion = [accumulated_pose[3], accumulated_pose[4], accumulated_pose[5], accumulated_pose[6]]
                incremental_pose_msg.pose.pose.orientation = Quaternion(*quaternion)
                vehicle_pose_pub.publish(incremental_pose_msg)

                text_marker = Marker()
                text_marker.header.frame_id = "base_link"
                text_marker.header.stamp = rospy.Time.now()
                text_marker.ns = "pose_text"
                text_marker.id = 0
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position = Point(accumulated_pose[0], accumulated_pose[1], accumulated_pose[2] + 1.0)  # Slightly above the vehicle
                text_marker.pose.orientation = Quaternion(0, 0, 0, 1)
                text_marker.scale.z = 0.5  # Height of the text
                text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # White color
                text_marker.text = f"Pos: ({accumulated_pose[0]:.2f}, {accumulated_pose[1]:.2f}, {accumulated_pose[2]:.2f})\n" \
                                   f"Quat: ({accumulated_pose[3]:.2f}, {accumulated_pose[4]:.2f}, {accumulated_pose[5]:.2f}, {accumulated_pose[6]:.2f})"
                pose_text_pub.publish(text_marker)

                """
                POINTCLOUD CLUSTERING PROCESS
                """
                rospy.loginfo("Pointcloud processing service")
                wc.pointcloud_processor.pointcloud_msg.header.frame_id = f"base_link_{scene_name}"
                wc.pointcloud_processor.pointcloud_msg.header.stamp = rospy.Time.now()
                wc.process_pointcloud()
                if wc.pointcloud.size < 0:
                    rospy.logerr("Pointcloud received is empty")
                    continue
                rospy.loginfo("Processed Pointcloud received")

                """
                LANDMARK DETECTION PROCESS
                """
                rospy.loginfo("Landmark detection processing service")
                wc.camera_processor.camera_msg.header.frame_id = f"base_link_{scene_name}"
                wc.camera_processor.camera_msg.header.stamp = rospy.Time.now()
                wc.process_image()
                if wc.processed_image is None:
                    rospy.logerr("Image received is empty")
                    continue

                """
                DATA ASSOCIATION > PROJECT POINTCLOUD ON IMAGE AND FILTER CLUSTERS
                This process 
                """
                rospy.loginfo("Data association process")
                # Pointcloud - Image projection
                wc.project_pointcloud_on_image()
                
                # Pointclouds filtering > get the IoU of pointcloud clusters and segmentation masks
                # This method generates a dictionary of the classes and those clusters that match the segmentations.
                # Generates clusters in the camera and vehicle frame
                wc.filter_association_iou(debug=False,iou_threshold=0.1)

                # DEBUG - Publish filtered pointcloud only with landmarks
                filtered_pointcloud_msg = PointCloud2()
                final_pointcloud = []
                for label, points in wc.clustered_pointcloud_iou_vehicle_frame.items():
                    for point in points:
                        final_pointcloud.append(point)
                
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1)
                ]
                header = Header()
                header.frame_id = "base_link"
                header.stamp = rospy.Time.now()
                filtered_pointcloud_msg = pc2.create_cloud(header, fields, final_pointcloud)
                filtered_pointcloud_pub.publish(filtered_pointcloud_msg)

                # Get cluster landmarks pose
                wc.calculate_landmark_pose()
                # TODO DEBUG - Publish landmark poses in vehicle and global frame

                frame_n += 1
                rate.sleep()
