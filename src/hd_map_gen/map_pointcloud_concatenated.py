"""
This script parses the maps in waymo dataset along with the 3D sem seg labels to generate and enrich the HD maps.
"""

import os
import sys

import numpy as np
import plotly.graph_objs as go
import pathlib
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()
import pcl

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *

def project_points_on_map(points):
    # Get pointcloud coordinated to project in the map
    xyz = points[0][:, 3:]
    num_points = xyz.shape[0]

    # Transform the points from the vehicle frame to the world frame.
    xyz = np.concatenate([xyz, np.ones([num_points, 1])], axis=-1)
    transform = np.reshape(np.array(frame.pose.transform), [4, 4])
    xyz = np.transpose(np.matmul(transform, np.transpose(xyz)))[:, 0:3]

    # Correct the pose of the points into the coordinate system of the first
    # frame to align with the map data.
    offset = frame.map_pose_offset
    points_offset = np.array([offset.x, offset.y, offset.z])
    xyz += points_offset

    return xyz


def get_color_palette_from_labels(labels):
    # Class Colors
    non_empty_labels = [arr for arr in labels if arr.size != 0]
    # Get only segmentation labels (not instance)
    labels = np.concatenate(non_empty_labels, axis=0)[:,1]

    return labels

def plot_pointcloud_on_map(map, point_cloud, point_cloud_labels):
    # Get color palette from labels to plot
    color_palette = get_color_palette_from_labels(point_cloud_labels)

    # Plot the point cloud for this frame aligned with the map data.
    figure = plot_maps.plot_map_features(map)

    figure.add_trace(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=color_palette,  # set color to an array/list of desired values
                colorscale='Pinkyl',  # choose a colorscale
                opacity=0.8,
            ),
        )
    )

    figure.show()


def segment_pointcloud(input):
    cloud = pcl.PointCloud()
    cloud.from_array(np.array(input.data, dtype=np.float32))

    u = [0.0, 0.0, 0.0]

    low_lim_x, low_lim_y, low_lim_z, up_lim_x, up_lim_y, up_lim_z = -0.5, -0.5, -0.5, 0.5, 0.5, 0.5

    cloudCropped = pcl.PointCloud()
    for point in cloud:
        if not (low_lim_x < point[0] < up_lim_x and low_lim_y < point[1] < up_lim_y and low_lim_z < point[2] < up_lim_z):
            cloudCropped.append(point)

    cloudCropped_np = np.asarray(cloudCropped)

    cloudCropped_np = cloudCropped_np.reshape((-1, 3))
    cloudCropped.from_array(cloudCropped_np.astype(np.float32))

    do_downsampling = True
    leafSize = 0.01

    if do_downsampling:
        vg = cloudCropped.make_voxel_grid_filter()
        vg.set_leaf_size(leafSize, leafSize, leafSize)
        cloudDownsampled = vg.filter()
    else:
        cloudDownsampled = cloudCropped

    msg_ds = pcl.PointCloud()
    msg_ds.from_array(np.asarray(cloudDownsampled))

    cloudNormals = cloudDownsampled.make_NormalEstimation()
    neGround = pcl.SACSegmentationFromNormals_PointXYZ_Normal()
    segGroundN = pcl.ModelCoefficients()
    inliersGround = pcl.PointIndices()

    neGround.setInputCloud(cloudDownsampled)
    neGround.setInputNormals(cloudNormals)
    neGround.setMethodType(0)
    neGround.setDistanceThreshold(0.1)
    neGround.setAxis(0, 0, 1)
    neGround.segment(inliersGround, segGroundN)

    cloudNoGround = cloudDownsampled.extract(inliersGround, negative=True)

    treeClusters = cloudNoGround.make_kdtree()
    ec = cloudNoGround.make_EuclideanClusterExtraction()
    clusterTolerance = 0.1
    ec.set_ClusterTolerance(clusterTolerance)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(25000)
    ec.set_SearchMethod(treeClusters)
    clusterIndices = ec.Extract()

    clustersTotal = pcl.PointCloud()
    n = 0

    for indices in clusterIndices:
        cloudCluster = pcl.PointCloud()
        for index in indices:
            cloudCluster.append(cloudNoGround[index])

        clustersTotal += cloudCluster

    msg_cl = pcl.PointCloud()
    msg_cl.from_array(np.asarray(clustersTotal))

    textScale = 0.1
    xpLine, ypLine, zpLine, xdLine, ydLine, zdLine = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    j, n = 0, 0

    segLineN = cloudNoGround.make_segmenter()
    segLineN.set_model_type(1)
    segLineN.set_method_type(0)
    segLineN.set_optimize_coefficients(True)
    segLineN.set_normal_distance_weight(0.1)
    segLineN.set_max_iterations(100)
    segLineN.set_distance_threshold(0.05)

    marker_aux = Marker()
    marker_aux.header.frame_id = "base_link"
    marker_aux.ns = "error_display"
    marker_aux.type = Marker.TEXT_VIEW_FACING
    marker_aux.action = Marker.ADD
    marker_aux.scale.z = textScale
    marker_aux.color.a = 1.0

    for indices in clusterIndices:
        cloudCluster = pcl.PointCloud()
        for index in indices:
            cloudCluster.append(cloudNoGround[index])

        cloudCluster_np = np.asarray(cloudCluster)
        cloudCluster_np = cloudCluster_np.reshape((-1, 3))
        cloudCluster.from_array(cloudCluster_np.astype(np.float32))

        cloudClusterNormals = cloudCluster.make_NormalEstimation()
        segLineN.setInputCloud(cloudCluster)
        segLineN.setInputNormals(cloudClusterNormals)

        inliersLine, coefficientsLine = segLineN.segment()

        xpLine, ypLine, zpLine, xdLine, ydLine, zdLine = coefficientsLine[:6]

        if len(inliersLine) != 0:
            if zdLine > math.cos(math.radians(10)):
                heightMax = 2.0
                errorMax, ratioMin = 0.05, 0.7

                zMax, zMin = -9.99e10, 9.99e10
                for point in cloudCluster:
                    if point[2] < zMin:
                        zMin, x_zMin, y_zMin = point[2], point[0], point[1]
                    if point[2] > zMax:
                        zMax, x_zMax, y_zMax = point[2], point[0], point[1]

                height = zMax - zMin

                distances = np.array(segLineN.get_distances())

                cumsum = np.sum(distances)
                cumsum2 = np.sum(distances ** 2)
                error = cumsum / len(inliersLine)
                ratio = len(inliersLine) / len(cloudCluster)

                isVerticalElement = error < errorMax and ratio > ratioMin and height < heightMax

                if isVerticalElement:
                    marker_aux.color.r, marker_aux.color.g, marker_aux.color.b = 0.0, 1.0, 0.0
                else:
                    marker_aux.color.r, marker_aux.color.g, marker_aux.color.b = 1.0, 0.0, 0.0

                marker_aux.id = n
                marker_aux.pose.position.x, marker_aux.pose.position.y, marker_aux.pose.position.z = xpLine, ypLine, zpLine + 2.0

                u[0], u[1], u[2] = -ydLine / math.sqrt(xdLine ** 2 + ydLine ** 2), xdLine / math.sqrt(xdLine ** 2 + yd)



if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        print("Scene {} processing: {}".format(str(scene_index), scene_path))

        # Array to store segmented pointclouds
        point_clouds = []
        # Array to store pointcloud labels
        point_cloud_labels = []

        for frame in load_frame(scene_path):
            # Get Map Information of the scene
            # For the first frame > Only retreive frame with map information
            # Save map features in a variable to use it with the semantic information from the LiDAR
            if hasattr(frame, 'map_features') and frame.map_features:
                # Retrieve map_feature in the firts frame
                map_features = frame.map_features

            print("Map feature found in scene")
            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            # Only generate information of 3D labeled frames
            if not(segmentation_labels):
                continue
            print("Segmentation label found")

            ## Get Lidar Pointcloud
            # Project the range images into points.
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                keep_polar_features=True,
            )
            print("PointCloud shape: ", np.shape(points))

            #  Get the labeled points from the point cloud
            point_labels = convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels)
            
            filtered_point_cloud, filtered_point_labels = filter_lidar_data(points, point_labels, [8, 10])
            print("PointCloud shape for only selected classes: ", np.shape(filtered_point_cloud))

            ## Project PointCloud on the Map
            projected_point_cloud = project_points_on_map(filtered_point_cloud)

            point_clouds.append(projected_point_cloud)
            point_cloud_labels.append(filtered_point_labels)

        # Concatenate pointclouds of the scene
        if len(point_clouds) > 1:
            concat_point_cloud = np.concatenate(point_clouds, axis=0)
            concat_point_cloud_labels = np.concatenate(point_cloud_labels, axis=0)
        else:
            print("No pointclouds in scene")
            continue

        ## Clustering of the PointCloud


        ## Plot Map and PointCloud aligned with the map data.
        plot_pointcloud_on_map(map_features, concat_point_cloud, concat_point_cloud_labels)



