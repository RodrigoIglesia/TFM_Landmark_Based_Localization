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
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import PoseStamped
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
from waymo_utils.waymo_3d_parser import *



class WaymoClient:
    def __init__(self, frame, cams_order):
        self.frame = frame
        self.cams_order = cams_order
        self.points = np.zeros((0,3))
        self.pointcloud = np.zeros((0, 3))
        self.cluster_labels = np.zeros((0, 3))
        self.clustered_pointcloud = {}
        self.clustered_pointcloud_image = {} # Dictionary class:cluster for image projected pointcloud
        self.init_camera_params()
        self.image = None # Original Image
        self.processed_image = None # Returned segmentation mask by the landmark detection service
        self.image_height = None
        self.image_width = None
        self.detection_association_image = None

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

    def project_pointcloud_on_image(self):
        """
        Returns: cluster labels dictionary, projected on camera frame --> {label: cluster}
        """
        if self.pointcloud.size == 0 or self.processed_image is None:
            rospy.logwarn("No pointcloud or processed image available for projection")
            return

        point_cloud_hom = self.cart2hom(self.pointcloud)
        point_cloud_cam = np.dot(point_cloud_hom, np.transpose(self.extrinsic_matrix_inv))

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


    def get_association_iou(self, min_hull_area = 500, iou_threshold=0.5):
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

                # Get cluster hull polygon
                cluster_poly = Polygon(cluster_hull.reshape(-1, 2))

                # Calculate IoU with all segmentation hulls
                for seg_hull in segmentation_hulls:
                    # Filter segmentation hulls by area
                    if (cv2.contourArea(seg_hull) <= min_hull_area):
                        continue
                    
                    # Get segmentation hull polygon
                    seg_poly = Polygon(seg_hull.reshape(-1,2))

                    ## TODO: REMOVE DEBUG
                    pair_image = np.copy(self.image)
                    cv2.drawContours(pair_image, [seg_hull.reshape((-1, 1, 2))], -1, (255, 0, 0), 2)  # White for segmentation hull
                    cv2.drawContours(pair_image, [cluster_hull.reshape((-1, 1, 2))], -1, (0, 0, 255), 2)  # Red for cluster hull
                    
                    iou = self.__calculate_iou(cluster_hull, seg_hull)
                    self.__draw_iou_text(pair_image, cluster_hull, seg_hull, iou)

                    # Show the image with the pair of hulls
                    cv2.imshow("Hull Pair", pair_image)
                    cv2.waitKey(0)
                    # if iou >= iou_threshold:
                    #     print("Intersection over Union: ", iou)
                    #     self.__draw_iou_text(black_image, cluster_hull, seg_hull, iou)

        # # Merge the black image with the processed image
        # self.detection_association_image = cv2.addWeighted(self.image, 1, black_image, 1, 0)


class IncrementalOdometryExtractor(WaymoClient):
    def __init__(self, frame):
        super().__init__(frame, None)
        self.prev_transform_matrix = None
        self.inc_odom_msg = PoseStamped()
        self.transform_matrix = np.array(self.frame.pose.transform).reshape(4, 4)

    def quaternion_to_euler(self, w, x, y, z):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def ominus(self, T1, T2):
        return np.linalg.inv(T2) @ T1

    def get_pose(self, T):
        position = T[:3, 3]
        R_matrix = T[:3, :3]
        rotation = R.from_matrix(R_matrix)
        quaternion = rotation.as_quat()
        roll, pitch, yaw = self.quaternion_to_euler(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
        return [position[0], position[1], position[2], roll, pitch, yaw]

    def process_odometry(self):
        if self.prev_transform_matrix is not None:
            inc_odom_matrix = self.ominus(self.transform_matrix, self.prev_transform_matrix)
            inc_odom = self.get_pose(inc_odom_matrix)

            # Create PoseStamped message
            self.inc_odom_msg.pose.position.x = inc_odom[0]
            self.inc_odom_msg.pose.position.y = inc_odom[1]
            self.inc_odom_msg.pose.position.z = inc_odom[2]
            quaternion = R.from_euler('xyz', [inc_odom[3], inc_odom[4], inc_odom[5]]).as_quat()
            self.inc_odom_msg.pose.orientation.x = quaternion[0]
            self.inc_odom_msg.pose.orientation.y = quaternion[1]
            self.inc_odom_msg.pose.orientation.z = quaternion[2]
            self.inc_odom_msg.pose.orientation.w = quaternion[3]
        
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

    rate = rospy.Rate(1/30)

    # Read dataset
    dataset_path = os.path.join(src_dir, "dataset/waymo_test_scene")
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
                wc.incremental_odometry = IncrementalOdometryExtractor(frame)

                rospy.loginfo("Calling processing services")

                rospy.loginfo("Pointcloud processing service")
                wc.pointcloud_processor.pointcloud_msg.header.frame_id = f"base_link_{scene_name}"
                wc.pointcloud_processor.pointcloud_msg.header.stamp = rospy.Time.now()
                wc.process_pointcloud()
                if wc.pointcloud.size < 0:
                    rospy.logerr("Pointcloud received is empty")
                    continue
                rospy.loginfo("Processed Pointcloud received")

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

                # Pointcloud - Image projection
                wc.project_pointcloud_on_image()

                # Pointclous filtering
                wc.get_association_iou()

                # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                # cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # cv2.imshow('result', wc.detection_association_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Process odometry
                # wc.incremental_odometry.process_odometry()

                rate.sleep()
