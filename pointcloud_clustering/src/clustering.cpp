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
#include <ros/publisher.h>
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
#include <boost/program_options.hpp>
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

// Configuration structure
struct Config {
    float low_lim_x, low_lim_y, low_lim_z, up_lim_x, up_lim_y, up_lim_z;
    int minPointsVoxel;
    float KSearchGround;
    bool OptimizeCoefficientsGround;
    float NormalDistanceWeightGround;
    float MaxIterationsGround;
    float DistanceThresholdGround;
    float clusterTolerance;
    bool do_cropping;
    bool do_downsampling;
    float leafSize;
    int clusterMinSize;
    int clusterMaxSize;
};

// PointCloudProcessor class
class PointCloudProcessor {
public:
    PointCloudProcessor(const std::string& configFilePath);
    bool processPointCloudService(pointcloud_clustering::clustering_srv::Request &req,
                                  pointcloud_clustering::clustering_srv::Response &res);
private:
    void readConfig(const std::string &filename);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudExtractGround(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr euclideanClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    void generatePointcloudMsg(pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> &cloud, sensor_msgs::PointCloud2 &cloudMsg);

    Config config_;
    std::mutex mtx_;
    std::string frame_id_;

    // Debug publishers -> publisher to debug processing results in RVIZ topics
    ros::Publisher input_pub_;
    ros::Publisher crop_pub_;
    ros::Publisher downsample_pub_;
    ros::Publisher ground_extract_pub_;
    ros::Publisher clustering_pub_;
};

PointCloudProcessor::PointCloudProcessor(const std::string& configFilePath) {
    ros::NodeHandle nh;
    input_pub_ = nh.advertise<sensor_msgs::PointCloud2>("input_pointcloud", 1);
    crop_pub_ = nh.advertise<sensor_msgs::PointCloud2>("cropped_pointcloud", 1);
    downsample_pub_ = nh.advertise<sensor_msgs::PointCloud2>("downsampled_pointcloud", 1);
    ground_extract_pub_ = nh.advertise<sensor_msgs::PointCloud2>("ground_extracted_pointcloud", 1);
    clustering_pub_ = nh.advertise<sensor_msgs::PointCloud2>("clustered_pointcloud", 1);
    readConfig(configFilePath);
}

void PointCloudProcessor::readConfig(const std::string &filename) {
    namespace po = boost::program_options;
    po::options_description config("Configuration");
    config.add_options()
        ("cropping.do_cropping", po::value<bool>(&config_.do_cropping)->default_value(true), "Cropping flag")
        ("cropping.low_lim_x", po::value<float>(&config_.low_lim_x)->default_value(-10.0), "Lower limit for x axis")
        ("cropping.low_lim_y", po::value<float>(&config_.low_lim_y)->default_value(-10.0), "Lower limit for y axis")
        ("cropping.low_lim_z", po::value<float>(&config_.low_lim_z)->default_value(-2.0), "Lower limit for z axis")
        ("cropping.up_lim_x", po::value<float>(&config_.up_lim_x)->default_value(10.0), "Upper limit for x axis")
        ("cropping.up_lim_y", po::value<float>(&config_.up_lim_y)->default_value(10.0), "Upper limit for y axis")
        ("cropping.up_lim_z", po::value<float>(&config_.up_lim_z)->default_value(2.0), "Upper limit for z axis")
        ("downsampling.do_downsampling", po::value<bool>(&config_.do_downsampling)->default_value(true), "Downsampling flag")
        ("downsampling.minPointsVoxel", po::value<int>(&config_.minPointsVoxel)->default_value(1), "Minimum points per voxel")
        ("downsampling.leafSize", po::value<float>(&config_.leafSize)->default_value(0.2), "Leaf size for downsampling")
        ("ground_extraction.KSearchGround", po::value<float>(&config_.KSearchGround)->default_value(50), "K search for ground extraction")
        ("ground_extraction.OptimizeCoefficientsGround", po::value<bool>(&config_.OptimizeCoefficientsGround)->default_value(true), "Optimize coefficients for ground extraction")
        ("ground_extraction.NormalDistanceWeightGround", po::value<float>(&config_.NormalDistanceWeightGround)->default_value(0.1), "Normal distance weight for ground extraction")
        ("ground_extraction.MaxIterationsGround", po::value<float>(&config_.MaxIterationsGround)->default_value(1000), "Max iterations for ground extraction")
        ("ground_extraction.DistanceThresholdGround", po::value<float>(&config_.DistanceThresholdGround)->default_value(0.05), "Distance threshold for ground extraction")
        ("clustering.clusterTolerance", po::value<float>(&config_.clusterTolerance)->default_value(0.02), "Cluster tolerance for euclidean clustering")
        ("clustering.clusterMinSize", po::value<int>(&config_.clusterMinSize)->default_value(100), "Cluster minimum points number")
        ("clustering.clusterMaxSize", po::value<int>(&config_.clusterMaxSize)->default_value(1000), "Cluster maximum points number");

    po::variables_map vm;
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    po::store(po::parse_config_file(ifs, config), vm);
    po::notify(vm);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudProcessor::pointcloudCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_DEBUG("Cropping received pointcloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropCloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (int k = 0; k < inputCloud->points.size(); k++)
    {
        if (inputCloud->points[k].x < config_.low_lim_x || inputCloud->points[k].x > config_.up_lim_x ||
            inputCloud->points[k].y < config_.low_lim_y || inputCloud->points[k].y > config_.up_lim_y ||
            inputCloud->points[k].z < config_.low_lim_z || inputCloud->points[k].z > config_.up_lim_z)
        {
            cropCloud->points.push_back(inputCloud->points[k]);
        }
    }

    cropCloud->width = cropCloud->points.size();
    cropCloud->height = 1;
    cropCloud->is_dense = true;

    // Debug publish
    sensor_msgs::PointCloud2 cropCloudMsg;
    pcl::toROSMsg(*cropCloud, cropCloudMsg);
    cropCloudMsg.header.frame_id = "base_link";
    cropCloudMsg.header.stamp = ros::Time::now();
    crop_pub_.publish(cropCloudMsg);

    return cropCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudProcessor::pointcloudDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_DEBUG("Downsampling received pointcloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr dsCloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(inputCloud);
    vg.setLeafSize(config_.leafSize, config_.leafSize, config_.leafSize);
    vg.setMinimumPointsNumberPerVoxel(config_.minPointsVoxel);
    vg.filter(*dsCloud);

    // Debug publish
    sensor_msgs::PointCloud2 dsCloudMsg;
    pcl::toROSMsg(*dsCloud, dsCloudMsg);
    dsCloudMsg.header.frame_id = "base_link";
    dsCloudMsg.header.stamp = ros::Time::now();
    downsample_pub_.publish(dsCloudMsg);

    return dsCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudProcessor::pointcloudExtractGround(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_DEBUG("Extracting ground from received pointcloud");
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
    neGround.setKSearch(config_.KSearchGround);
    neGround.compute(*cloudNormals);

    segGroundN.setOptimizeCoefficients(config_.OptimizeCoefficientsGround);
    segGroundN.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    segGroundN.setNormalDistanceWeight(config_.NormalDistanceWeightGround);
    segGroundN.setMethodType(pcl::SAC_RANSAC);
    segGroundN.setMaxIterations(config_.MaxIterationsGround);
    segGroundN.setDistanceThreshold(config_.DistanceThresholdGround);
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

    // Debug publish
    sensor_msgs::PointCloud2 groundExtractMsg;
    pcl::toROSMsg(*cloudNoGround, groundExtractMsg);
    groundExtractMsg.header.frame_id = "base_link";
    groundExtractMsg.header.stamp = ros::Time::now();
    ground_extract_pub_.publish(groundExtractMsg);

    return cloudNoGround;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcessor::euclideanClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    ROS_DEBUG("Clustering received pointcloud");
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr treeClusters(new pcl::search::KdTree<pcl::PointXYZ>);
    treeClusters->setInputCloud(inputCloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    ec.setClusterTolerance(config_.clusterTolerance);

    // float clusterMinSize, clusterMaxSize;
    // clusterMinSize = -25.0 * config_.leafSize + 17.5;
    // clusterMaxSize = -500.0 * config_.leafSize + 350.0;


    ec.setMinClusterSize(config_.clusterMinSize);
    ec.setMaxClusterSize(config_.clusterMaxSize);
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

    // Debug publish
    sensor_msgs::PointCloud2 clusterCloudMsg;
    pcl::toROSMsg(*clusteredCloud, clusterCloudMsg);
    clusterCloudMsg.header.frame_id = "base_link";
    clusterCloudMsg.header.stamp = ros::Time::now();
    clustering_pub_.publish(clusterCloudMsg);

    return clusteredCloud;
}

void PointCloudProcessor::generatePointcloudMsg(pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> &cloud, sensor_msgs::PointCloud2 &cloudMsg)
{
    pcl::toROSMsg(*cloud, cloudMsg);
    cloudMsg.header.frame_id = "base_link";
}

bool PointCloudProcessor::processPointCloudService(pointcloud_clustering::clustering_srv::Request &req,
                                                   pointcloud_clustering::clustering_srv::Response &res)
{
    std::lock_guard<std::mutex> lock(mtx_);
    frame_id_ = req.pointcloud.header.frame_id;
    ROS_DEBUG("PointCloud received: %s", frame_id_.c_str());

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCropped(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDownsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudGroundExtracted(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudClustered(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::fromROSMsg(req.pointcloud, *cloud);

    // debug publishing
    sensor_msgs::PointCloud2 inputCloudMsg;
    pcl::toROSMsg(*cloud, inputCloudMsg);
    inputCloudMsg.header.frame_id = "base_link";
    inputCloudMsg.header.stamp = ros::Time::now();
    input_pub_.publish(inputCloudMsg);

    if (config_.do_cropping)
    {
        cloudCropped = pointcloudCrop(cloud);
    }
    else
    {
        cloudCropped = cloud;
    }
    

    if (config_.do_downsampling)
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

    ROS_DEBUG("Server response completed");

    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clustering_service");
    ros::NodeHandle nh;

    // Current working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        ROS_DEBUG("Current working directory: %s", cwd);
    } else {
        ROS_ERROR("Error getting current working directory");
    }

    std::string configFilePath;
    if (!nh.getParam("config_file_path", configFilePath)) {
        ROS_ERROR("Failed to get param 'config_file_path'");
        return -1;
    }
    try {
        PointCloudProcessor processor(configFilePath);
        ros::ServiceServer service = nh.advertiseService("process_pointcloud", &PointCloudProcessor::processPointCloudService, &processor);
        ROS_DEBUG("Service clustering initialized correctly");

        ros::spin();
    } catch (const std::exception &e) {
        ROS_ERROR("Error initializing PointCloudProcessor: %s", e.what());
        return -1;
    }

    return 0;
}