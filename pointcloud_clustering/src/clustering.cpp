/*
    Author: Rodrigo de la Iglesia Sánchez (2024)
    Euclidean clustering Program
    Source: https://github.com/ahermosin/TFM_velodyne (Alberto Hermosín, 2019)
    This program is implemented as a ROS Service instead of a ROS node.
    Applies an euclidean clustering algorithm to a concatenated pointcloud, in order to extract the clusters for vertical elements.
    Output of this process is a clustered pointcloud.
*/

#include <ros/ros.h>
#include <rosbag/bag.h>
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
#include <mutex>
#include "pointcloud_clustering/clustering_srv.h"

// Declare a mutex for thread-safe operations
std::mutex mtx;

float leafSize = 0.2;
std::string frame_id;

pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_INFO("Cropping received pointcloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropCloud(new pcl::PointCloud<pcl::PointXYZ>);
    float low_lim_x, low_lim_y, low_lim_z, up_lim_x, up_lim_y, up_lim_z;
    ros::param::get("low_lim_x", low_lim_x);
    ros::param::get("low_lim_y", low_lim_y);
    ros::param::get("low_lim_z", low_lim_z);
    ros::param::get("up_lim_x", up_lim_x);
    ros::param::get("up_lim_y", up_lim_y);
    ros::param::get("up_lim_z", up_lim_z);

    for (int k = 0; k < inputCloud->points.size(); k++)
    {
        if (inputCloud->points[k].x < low_lim_x || inputCloud->points[k].x > up_lim_x ||
            inputCloud->points[k].y < low_lim_y || inputCloud->points[k].y > up_lim_y ||
            inputCloud->points[k].z < low_lim_z || inputCloud->points[k].z > up_lim_z)
        {
            cropCloud->points.push_back(inputCloud->points[k]);
        }
    }

    cropCloud->width = cropCloud->points.size();
    cropCloud->height = 1;
    cropCloud->is_dense = true;

    return cropCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_INFO("Downsampling received pointcloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr dsCloud(new pcl::PointCloud<pcl::PointXYZ>);
    int minPointsVoxel;
    ros::param::get("minPointsVoxel", minPointsVoxel);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(inputCloud);
    vg.setLeafSize(leafSize, leafSize, leafSize);
    vg.setMinimumPointsNumberPerVoxel(minPointsVoxel);
    vg.filter(*dsCloud);

    return dsCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudExtractGround(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_INFO("Extracting ground from received pointcloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNoGround(new pcl::PointCloud<pcl::PointXYZ>());

    float theta;
    float u[3];
    tf::Quaternion tfQuat;

    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> neGround;
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> segGroundN;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr treeGround(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::ModelCoefficients::Ptr coefficientsGround(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliersGround(new pcl::PointIndices);

    neGround.setSearchMethod(treeGround);
    neGround.setInputCloud(inputCloud);
    float KSearchGround;
    ros::param::get("KSearchGround", KSearchGround);
    neGround.setKSearch(KSearchGround);
    neGround.compute(*cloudNormals);

    bool OptimizeCoefficientsGround;
    ros::param::get("OptimizeCoefficientsGround", OptimizeCoefficientsGround);
    segGroundN.setOptimizeCoefficients(OptimizeCoefficientsGround);
    segGroundN.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    float NormalDistanceWeightGround;
    ros::param::get("NormalDistanceWeightGround", NormalDistanceWeightGround);
    segGroundN.setNormalDistanceWeight(NormalDistanceWeightGround);
    segGroundN.setMethodType(pcl::SAC_RANSAC);
    float MaxIterationsGround;
    ros::param::get("MaxIterationsGround", MaxIterationsGround);
    segGroundN.setMaxIterations(MaxIterationsGround);
    float DistanceThresholdGround;
    ros::param::get("DistanceThresholdGround", DistanceThresholdGround);
    segGroundN.setDistanceThreshold(DistanceThresholdGround);
    segGroundN.setInputCloud(inputCloud);
    segGroundN.setInputNormals(cloudNormals);
    segGroundN.segment(*inliersGround, *coefficientsGround);

    if (coefficientsGround->values.size() != 0)
    {
        geometry_msgs::Vector3 groundDistance;
        geometry_msgs::Vector3 groundDirection;

        groundDirection.x = coefficientsGround->values[0];
        groundDirection.y = coefficientsGround->values[1];
        groundDirection.z = coefficientsGround->values[2];

        groundDistance.x = coefficientsGround->values[3] * coefficientsGround->values[0];
        groundDistance.y = coefficientsGround->values[3] * coefficientsGround->values[1];
        groundDistance.z = coefficientsGround->values[3] * coefficientsGround->values[2];

        u[0] = (-groundDirection.y) / (sqrt(pow(groundDirection.x, 2) + pow(groundDirection.y, 2)));
        u[1] = (groundDirection.x) / (sqrt(pow(groundDirection.x, 2) + pow(groundDirection.y, 2)));
        u[2] = 0.0;

        theta = acos(groundDirection.z);

        tfQuat = {sin(theta / 2) * u[0], sin(theta / 2) * u[1], sin(theta / 2) * u[2], cos(theta / 2)};

        extract.setInputCloud(inputCloud);
        extract.setIndices(inliersGround);
        extract.setNegative(true);
        extract.filter(*cloudNoGround);
    }
    else
    {
        *cloudNoGround = *inputCloud;
    }

    return cloudNoGround;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr euclideanClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_INFO("Clustering received pointcloud");
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr treeClusters(new pcl::search::KdTree<pcl::PointXYZ>);
    treeClusters->setInputCloud(inputCloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    float clusterTolerance;
    ros::param::get("clusterTolerance", clusterTolerance);

    ec.setClusterTolerance(clusterTolerance);

    float clusterMinSize, clusterMaxSize;
    clusterMinSize = -25.0 * leafSize + 17.5;
    clusterMaxSize = -500.0 * leafSize + 350.0;

    ec.setMinClusterSize(clusterMinSize);
    ec.setMaxClusterSize(clusterMaxSize);
    ec.setSearchMethod(treeClusters);
    ec.setInputCloud(inputCloud);
    ec.extract(clusterIndices);

    for (int i = 0; i < clusterIndices.size(); ++i)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointIndices cluster = clusterIndices[i];
        uint8_t r = rand() % 256;
        uint8_t g = rand() % 256;
        uint8_t b = rand() % 256;

        for (int j = 0; j < cluster.indices.size(); ++j)
        {
            pcl::PointXYZRGB point;
            point.x = inputCloud->points[cluster.indices[j]].x;
            point.y = inputCloud->points[cluster.indices[j]].y;
            point.z = inputCloud->points[cluster.indices[j]].z;
            point.r = r;
            point.g = g;
            point.b = b;
            cluster_cloud->points.push_back(point);
        }
        *clusteredCloud += *cluster_cloud;
    }

    return clusteredCloud;
}

void generatePointcloudMsg(pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> &cloud, sensor_msgs::PointCloud2 &cloudMsg)
{
    pcl::toROSMsg(*cloud, cloudMsg);
    cloudMsg.header.frame_id = "base_link";
}

bool processPointCloudService(pointcloud_clustering::clustering_srv::Request &req,
                              pointcloud_clustering::clustering_srv::Response &res)
{
    std::lock_guard<std::mutex> lock(mtx);
    frame_id = req.pointcloud.header.frame_id;
    ROS_INFO("PointCloud received: %s", frame_id.c_str());

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCropped(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDownsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudGroundExtracted(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudClustered(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::fromROSMsg(req.pointcloud, *cloud);

    cloudCropped = pointcloudCrop(cloud);
    float do_downsampling;
    ros::param::get("do_downsampling", do_downsampling);

    if (do_downsampling)
    {
        cloudDownsampled = pointcloudDownsample(cloudCropped);
    }
    else
    {
        cloudDownsampled = cloudCropped;
    }

    cloudGroundExtracted = pointcloudExtractGround(cloudDownsampled);
    cloudClustered = euclideanClustering(cloudGroundExtracted);

    pcl::toROSMsg(*cloudClustered, res.clustered_pointcloud);
    res.clustered_pointcloud.header.frame_id = "base_link";
    res.clustered_pointcloud.header.stamp = ros::Time::now();

    ROS_INFO("Server response completed");

    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clustering_service");
    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("process_pointcloud", processPointCloudService);
    ROS_INFO("Service server initialized correctly");

    ros::spin();

    return 0;
}
