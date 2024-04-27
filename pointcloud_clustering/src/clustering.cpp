/*
    Author: Rodrigo de la Iglesia Sánchez (2024)
    Euclidean clustering Program
    Source: https://github.com/ahermosin/TFM_velodyne (Alberto Hermosín, 2019)
    This program is implemented as a ROS Package and executed in a ROS node.
    Applies an euclidean clustering algorithm to a concatenated pointcloud, in order to extract the clusters for vertical elements.
    Output of this processes are clustered poinclouds, which will be feeded to an EKF (or similar algorithms) in further steps.
*/

#include <ros/ros.h>
#include <time.h>
#include <math.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/LinearMath/Quaternion.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/foreach.hpp>
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Header.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_ros/transforms.h>
#include <stdexcept>

#include <iostream>
#include <fstream>


// Prepare ROS message for publishing the clustered point cloud
sensor_msgs::PointCloud2 output_cloud;

void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    ROS_INFO("Subscibed to Test PointCloud topic correctle");
    // Convert ROS PointCloud2 message to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    pcl::toROSMsg(*cloud, output_cloud);
    output_cloud.header.stamp = ros::Time::now();
    output_cloud.header.frame_id = "base_link";
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clustering");
    ROS_INFO("Node clustering initialized correctly");
    ros::NodeHandle nh;
    // Subscribe to the Test_PointCloud topic
    ros::Subscriber sub = nh.subscribe("Test_PointCloud", 10, pointcloudCallback);
    // Publish the clustered point cloud
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("clustered_PointCloud", 10);

    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        ros::spinOnce();

        pub.publish(output_cloud);
        ROS_INFO("Point cloud published");

        loop_rate.sleep();
    }

    return (0);
}