#!/usr/bin/env python

import rospy
from visualization_msgs.msg import MarkerArray, Marker
import tf
import math
import numpy as np
import pcl
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Vector3
from pcl import pcl_visualization, PointXYZ, Normal, SACSegmentationFromNormals, ModelCoefficients, PointIndices, ExtractIndices, EuclideanClusterExtraction, VoxelGrid, search


def callback(input):
    t1 = rospy.Time.now()

    clusters = PointCloud2()
    msg_aux = PoseArray()
    msg_aux_BL = PoseArray()
    marker_array_aux = MarkerArray()

    cloud = pcl.PointCloud()
    cloudCropped = pcl.PointCloud()
    cloudDownsampled = pcl.PointCloud()

    # Convert ROS PointCloud2 to PCL
    points = pcl.PointCloud_PointXYZ()
    pcl.fromROSMsg(input, points)
    cloud.from_array(points.to_array())

    pose = PoseStamped() # For quaternion conversions
    tfQuat = tf.Quaternion()
    gmQuat = Quaternion() 
    theta = 0.0
    u = [0.0, 0.0, 0.0] # u is the normalized vector used to turn Z axis until it reaches the orientation of the ground

    # Cropping
    cloudCropped = cloud.make_passthrough_filter()
    cloudCropped.set_filter_field_name("x")
    cloudCropped.set_filter_limits(-0.5, 0.5)
    cloudCropped.filter(cloudCropped)
    cloudCropped.set_filter_field_name("y")
    cloudCropped.set_filter_limits(-0.5, 0.5)
    cloudCropped.filter(cloudCropped)
    cloudCropped.set_filter_field_name("z")
    cloudCropped.set_filter_limits(-0.5, 0.5)
    cloudCropped.filter(cloudCropped)

    cloudCropped = cloud.extract(cloudCropped)

    # Downsample
    leafSize = 0.01
    do_downsampling = True
    if do_downsampling:
        vg = cloudCropped.make_voxel_grid_filter()
        vg.set_leaf_size(leafSize, leafSize, leafSize)
        vg.filter(cloudDownsampled)
    else:
        cloudDownsampled = cloudCropped

    # Estimate normals
    cloudNormals = cloudDownsampled.make_NormalEstimation()
    neGround = SACSegmentationFromNormals()
    segGroundN = SACSegmentationFromNormals()
    extract = ExtractIndices()
    treeGround = search.KdTree()

    neGround.set_search_method(treeGround)
    neGround.set_input_cloud(cloudDownsampled)
    KSearchGround = 100
    neGround.set_k_search(KSearchGround)
    cloudNormals = neGround.compute()

    # Ground extraction
    cloudNoGround = pcl.PointCloud()
    segGroundN.set_optimize_coefficients(True)
    segGroundN.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    segGroundN.set_method_type(pcl.SAC_RANSAC)
    segGroundN.set_normal_distance_weight(0.1)
    segGroundN.set_max_iterations(100)
    segGroundN.set_distance_threshold(0.2)
    segGroundN.set_input_cloud(cloudDownsampled)
    segGroundN.set_input_normals(cloudNormals)
    inliersGround, coefficientsGround = segGroundN.segment()

    if coefficientsGround.size() != 0:
        # Do something
        pass
    else:
        cloudNoGround = cloudDownsampled

    # Euclidean clustering
    treeClusters = search.KdTree()
    treeClusters.set_input_cloud(cloudNoGround)
    clusterIndices = PointIndices()
    ec = EuclideanClusterExtraction()
    clusterTolerance = 0.02
    
    ec.set_cluster_tolerance(clusterTolerance)
    ec.set_min_cluster_size(100)
    ec.set_max_cluster_size(25000)
    ec.set_search_method(treeClusters)
    ec.set_input_cloud(cloudNoGround)
    ec.extract(clusterIndices)

    # Cluster analysis
    clustersTotal = pcl.PointCloud()
    for cluster in clusterIndices:
        cloudCluster = pcl.PointCloud()
        for idx in cluster.indices:
            cloudCluster.points.append(cloudNoGround.points[idx])

        clustersTotal += cloudCluster

    # Prepare ROS messages
    msg_cl = clustersTotal.toROSMsg()
    msg_cl.header.stamp = rospy.Time.now()
    msg_cl.header.frame_id = "base_link"

    msg_ve = msg_aux
    msg_ve_BL = msg_aux_BL
    marker_array = marker_array_aux

    t2 = rospy.Time.now()
    rospy.loginfo("Time: %f", (t2-t1).to_sec()*1000)

def main():
    rospy.init_node('euclidean_clustering')
    subscription = rospy.get_param('~subscription', '/input_topic')
    rospy.Subscriber(subscription, PointCloud2, callback)
    pub_ds = rospy.Publisher('PointCloud2_ds', PointCloud2, queue_size=1)
    pub_ground = rospy.Publisher('poseGround', PoseStamped, queue_size=1)
    pub_ng = rospy.Publisher('PointCloud2_ng', PointCloud2, queue_size=1)
    pub_cl = rospy.Publisher('PointCloud2_cl', PointCloud2, queue_size=1)
    pub_ve = rospy.Publisher('PoseArray_ve', PoseArray, queue_size=1)
    pub_ve_BL = rospy.Publisher('PoseArray_ve_BL', PoseArray, queue_size=1)
    pub_text = rospy.Publisher('MarkerArray_text', MarkerArray, queue_size=1)
    listener = tf.TransformListener()
    logfile = rospy.get_param('~logfile', False)
    if logfile:
        myfile = open("/home/alberto/workspaces/workspace14diciembre/logVE.txt", "w")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform("map", "base_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        # Publish messages here

        rate.sleep()

if __name__ == '__main__':
    main()
