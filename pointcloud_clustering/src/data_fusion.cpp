//  Created on: Jul 29, 2013
//      Author: pdelapuente

#include <ros/ros.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <tf/tf.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include <geometry_msgs/PoseArray.h>
#include <pointcloud_clustering/positionRPY.h>
#include <pointcloud_clustering/observationRPY.h>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "custom_functions.h"

// Configuration structure
struct Config {
    float x_init;
};

class DataFusion {
public:
    DataFusion(const std::string& configFilePath);
    bool dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req,
                                  pointcloud_clustering::data_fusion_srv::Response &res);
private:
    void readConfig(const std::string &filename);
    // Position correction method
    pointcloud_clustering::positionRPY CorrectPosition()
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudExtractGround(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr euclideanClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
    void generatePointcloudMsg(pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> &cloud, sensor_msgs::PointCloud2 &cloudMsg);

    Config config_;
    std::string frame_id_;
};

DataFusion::DataFusion(const std::string& configFilePath) {
    // Initialization method -> create a node handler and read configuration
    ros::NodeHandle nh;
    readConfig(configFilePath);
}

void PointCloudProcessor::readConfig(const std::string &filename) {
    namespace po = boost::program_options;
    po::options_description config("Configuration");
    config.add_options()
    ("data_fusion.x_init", po::value<float>(&config_.x_init)->default_value(0.0), "Initial X")
    po::variables_map vm;
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    po::store(po::parse_config_file(ifs, config), vm);
    po::notify(vm);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "data_fusion_service");
    ros::NodeHandle nh;

    // Current working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        ROS_INFO("Current working directory: %s", cwd);
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
        ros::ServiceServer service = nh.advertiseService("data_fusion", &PointCloudProcessor::processPointCloudService, &processor);
        ROS_INFO("Service data fusion initialized correctly");

        ros::spin();
    } catch (const std::exception &e) {
        ROS_ERROR("Error initializing PointCloudProcessor: %s", e.what());
        return -1;
    }

    return 0;
}