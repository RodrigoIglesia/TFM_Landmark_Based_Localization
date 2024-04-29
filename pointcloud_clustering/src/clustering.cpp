/*
    Author: Rodrigo de la Iglesia Sánchez (2024)
    Euclidean clustering Program
    Source: https://github.com/ahermosin/TFM_velodyne (Alberto Hermosín, 2019)
    This program is implemented as a ROS Package and executed in a ROS node.
    Applies an euclidean clustering algorithm to a concatenated pointcloud, in order to extract the clusters for vertical elements.
    Output of this processes are clustered poinclouds, which will be feeded to an EKF (or similar algorithms) in further steps.
*/


//TODO: Crear clase clustering
//TODO: Crear un fichero de configuración global (no solo para ROS) y acceder desde el código


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


// Flag that determined that a point cloud has been published as input > callback entered
bool receivedPointCloud = false;

// Output messages declaration
//TODO: Estas variables deben ser atributos de la clase clustering
sensor_msgs::PointCloud2 inCloudMsg; // Input message -> debug
sensor_msgs::PointCloud2 cropCloudMsg; // Pointcloud cropped -> debug
sensor_msgs::PointCloud2 dsCloudMsg; // Pointcloud downsampled -> debug
sensor_msgs::PointCloud2 geCloudMsg; // Pointcloud ground extracted -> debug

pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    /*
    Given an input pointcloud apply cropping to extract the invalid points
    Invalid points are those that belongs inside a minimum dimensions cube
    */
   // Output cloud after cropping.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropCloud(new pcl::PointCloud<pcl::PointXYZ>);
    float low_lim_x, low_lim_y, low_lim_z, up_lim_x, up_lim_y, up_lim_z; // So it will be done manually
    ros::param::get("low_lim_x", low_lim_x);
    ros::param::get("low_lim_y", low_lim_y);
    ros::param::get("low_lim_z", low_lim_z);
    ros::param::get("up_lim_x", up_lim_x);
    ros::param::get("up_lim_y", up_lim_y);
    ros::param::get("up_lim_z", up_lim_z);

    int k = 0;
    for (k = 0; k <= inputCloud->points.size(); k++) // Remove the points within a cube of volume (up_lim_x - low_lim_x)m x (up_lim_y - low_lim_y)m x (up_lim_z - low_lim_z)m centered in the sensor. In order to avoid 6 comparations at a time it will be performed in three steps, saving efforts
    {
        if (inputCloud->points[k].x < low_lim_x || inputCloud->points[k].x > up_lim_x)
            cropCloud->points.push_back(inputCloud->points[k]);
        else if (inputCloud->points[k].y < low_lim_y || inputCloud->points[k].y > up_lim_y)
            cropCloud->points.push_back(inputCloud->points[k]);
        else if (inputCloud->points[k].z < low_lim_z || inputCloud->points[k].z > up_lim_z)
            cropCloud->points.push_back(inputCloud->points[k]);
    }

    cropCloud->width = cropCloud->points.size();
    cropCloud->height = 1;
    cropCloud->is_dense = true;

    return cropCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    /*
    Given an input pointcloud apply downsampling to reduce its density
    */
    // Output cloud after downsampling.
    pcl::PointCloud<pcl::PointXYZ>::Ptr dsCloud(new pcl::PointCloud<pcl::PointXYZ>);
    float leafSize;

    int minPointsVoxel;
    ros::param::get("minPointsVoxel", minPointsVoxel);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(inputCloud);
    ros::param::get("leafSize", leafSize);
    vg.setLeafSize(leafSize, leafSize, leafSize);
    vg.setMinimumPointsNumberPerVoxel(minPointsVoxel);
    vg.filter(*dsCloud);

    return dsCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloudExtractGround(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud)
{
    /*
    Given an input pointcloud apply downsampling to reduce its density
    */
    // Output cloud after extracting the ground.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNoGround(new pcl::PointCloud<pcl::PointXYZ>());

    //TODO: Estas variables tienen que ser atributos de la clase Clustering

    float theta;
    // u is the normalized vector used to turn Z axis until it reaches the orientation of the ground, by turning theta radians. It is obtained by cross-multiplying Z-axis and the director vector of the vertical element
    float u[3];
    tf::Quaternion tfQuat;

    // Normalize the imput pointcloud
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> neGround, neClusters;
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> segGroundN;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr treeGround(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::ModelCoefficients::Ptr coefficientsGround(new pcl::ModelCoefficients);
    pcl::ModelCoefficients::Ptr coefficientsLine(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliersGround(new pcl::PointIndices), inliersLine(new pcl::PointIndices);

    neGround.setSearchMethod(treeGround);
    neGround.setInputCloud(inputCloud); // Doing downsampling increases computation speed up to 10 times
    float KSearchGround;
    ros::param::get("KSearchGround", KSearchGround);
    neGround.setKSearch(KSearchGround); // The higher KSearchGround is, the slower neGround.compute gets
    neGround.compute(*cloudNormals);

    // Extract the ground from the pointcloud

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

        tfQuat = {sin(theta / 2) * u[0], sin(theta / 2) * u[1], sin(theta / 2) * u[2], cos(theta / 2)}; // We need to translate this information into quaternion form

        extract.setInputCloud(inputCloud); // Extract the planar inliers from the input cloud
        extract.setIndices(inliersGround);
        extract.setNegative(true);
        extract.filter(*cloudNoGround);
    }
    else
        *cloudNoGround = *inputCloud;

    return cloudNoGround;
}

void generatePointcloudMsg(pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> &cloud, sensor_msgs::PointCloud2 &cloudMsg)
{
    // Generate message for a pointcloud
    pcl::toROSMsg(*cloud, cloudMsg);
    cloudMsg.header.stamp = ros::Time::now();
    cloudMsg.header.frame_id = "base_link";
}

void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    ROS_INFO("Subscibed to Test PointCloud topic correctle");

    // Poin Cloud variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); // Received pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCropped(new pcl::PointCloud<pcl::PointXYZ>); // Pointcloud cropped > invalid points extracted
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDownsampled(new pcl::PointCloud<pcl::PointXYZ>); // Pointcloud downsampled > density reduce
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudGroundExtracted(new pcl::PointCloud<pcl::PointXYZ>); // Pointcloud without ground

    // Convert ROS PointCloud2 message to PCL point cloud
    pcl::fromROSMsg(*msg, *cloud);
    // Generate message for the imput pointcloud
    generatePointcloudMsg(cloud, inCloudMsg);

    /*
    Crop > delete points near to the sensors
    */
    cloudCropped = pointcloudCrop(cloud);
    // Generate message for the cropped pointcloud
    generatePointcloudMsg(cloudCropped, cropCloudMsg);

    /*
    Downsample > Reduce the density of the pointcloud
    */
    float do_downsampling;;
    ros::param::get("do_downsampling", do_downsampling);

    if (do_downsampling == true)
    {
        cloudDownsampled = pointcloudDownsample(cloudCropped);
    }
    else
        cloudDownsampled = cloudCropped;
    // Generate message for the downsampled pointcloud
    generatePointcloudMsg(cloudDownsampled, dsCloudMsg);

    /*
    Extract Ground
    */
    cloudGroundExtracted = pointcloudExtractGround(cloudDownsampled);
    // Generate message for the downsampled pointcloud
    generatePointcloudMsg(cloudGroundExtracted, geCloudMsg);

    // Callback has been entered
    receivedPointCloud = true;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "clustering");
    ROS_INFO("Node clustering initialized correctly");
    ros::NodeHandle nh;
    // Subscribe to the Test_PointCloud topic
    ros::Subscriber sub = nh.subscribe("Test_PointCloud", 10, pointcloudCallback);

    // Publishing topics
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("input_PointCloud", 10); // input pointcloud
    ros::Publisher pub_crop = nh.advertise<sensor_msgs::PointCloud2>("cropped_PointCloud", 10); // cropped pointcloud
    ros::Publisher pub_ds = nh.advertise<sensor_msgs::PointCloud2>("ds_PointCloud", 10); // cropped pointcloud
    ros::Publisher pub_ge = nh.advertise<sensor_msgs::PointCloud2>("ge_PointCloud", 10); // Ground extracted pointcloud

    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        ros::spinOnce();

        if (receivedPointCloud)
        {
            pub.publish(inCloudMsg);
            ROS_INFO("Point cloud published");
            pub_crop.publish(cropCloudMsg);
            ROS_INFO("Point cloud Cropped published");
            pub_ds.publish(dsCloudMsg);
            ROS_INFO("Point cloud Downsampled published");
            pub_ge.publish(geCloudMsg);
            ROS_INFO("Point cloud without ground published");
        }

        loop_rate.sleep();
    }

    return (0);
}