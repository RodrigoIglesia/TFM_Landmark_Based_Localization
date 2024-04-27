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

class PointCloudSubscriber
{
public:
    PointCloudSubscriber()
    {
        // Subscribe to the point cloud topic
        pointcloud_sub_ = nh_.subscribe("Test_PointCloud", 1, &PointCloudSubscriber::pointCloudCallback, this);

        // Advertise the clustered point cloud topic
        clustered_pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_PointCloud", 1);
    }

    // Point cloud callback function
    void pointCloudCallback()
    {
        // Publish the received point cloud to the clustered_PointCloud topic
        clustered_pointcloud_pub_.publish(point_cloud_msg);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher clustered_pointcloud_pub_;
    // Subscribed PointCloud message
    sensor_msgs::PointCloud2 point_cloud_msg;
};

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "pointcloud_subscriber_cpp");

    // Create an instance of the PointCloudSubscriber class
    PointCloudSubscriber pointCloudSubscriber;

    // Spin
    ros::spin();

    return 0;
}