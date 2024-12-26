//  Created on: Jul 29, 2013
//      Author: pdelapuente
//      Updated (Dec 01, 2024): Rodrigo de la Iglesia

//TODO: Comprobar si las observaciones en el sistema de referencia del vehículo son necesitadas
//TODO: Cambiar observations por observations_BL y hacer transformaciones a global frame
//TODO: De momento el EKF frame es referenciado a 0,0,0 como pose inicial del vehículo
//TODO: comprobar que los incrementos de odometría recibidos son respecto a la pose anterior, no respecto a la primera pose, si son respecto a la primera pose ya es la pose referenciada del frame del EKF

#include <ros/ros.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <tf/tf.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <boost/program_options.hpp>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include <geometry_msgs/PoseArray.h>
#include <pointcloud_clustering/positionRPY.h>
#include <pointcloud_clustering/observationRPY.h>
#include <pointcloud_clustering/data_fusion_srv.h>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include "custom_functions.h"

using namespace Eigen;
typedef Eigen::Matrix<float, 5, 5> Matrix5f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 5, 1> Vector5f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;

struct Config
{
    float x_init;
    float y_init;
    float z_init;
    float roll_init;
    float pitch_init;
    float yaw_init;
    float sigma_odom_x;
    float sigma_odom_y;
    float sigma_odom_z;
    float sigma_odom_roll;
    float sigma_odom_pitch;
    float sigma_odom_yaw;
    float sigma_obs_x;
    float sigma_obs_y;
    float sigma_obs_z;
    float sigma_obs_roll;
    float sigma_obs_pitch;
    float sigma_obs_yaw;
    float mahalanobisDistanceThreshold;
    float QFactor;
    float P00_init;
    float P11_init;
    float P22_init;
    float P33_init;
    float P44_init;
    float P55_init;
    float PFactor;
};

class DataFusion
{
public:
    DataFusion(const std::string& configFilePath, const std::string& mapFilePath);
    bool dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req, pointcloud_clustering::data_fusion_srv::Response &res);
    std::vector<pointcloud_clustering::positionRPY> processPoseArray(const geometry_msgs::PoseArray& poseArray);
    std::vector<pointcloud_clustering::positionRPY> loadMapFromCSV();
    void publishPoseElements(std::vector<pointcloud_clustering::positionRPY> pose_array, ros::Publisher publisher);

private:
    void readConfig(const std::string &filename);
    Config config_;
    std::string mapFilePath;
    int mapSize;

    int frame_n;

    // Member variables to store the state across service calls
    pointcloud_clustering::positionRPY kalmanPose; // Kalman corrected pose
    Matrix6f P;  // Covariance matrix
    Matrix6f Q;  // Process noise covariance
    Matrix6f R;  // Observation noise covariance
    float PFactor;  // Factor for the covariance matrix scaling
    float QFactor;  // Factor for the covariance matrix scaling
    pointcloud_clustering::positionRPY sigma_odom;
    pointcloud_clustering::positionRPY sigma_obs;
    std::vector<pointcloud_clustering::positionRPY> map;

    // Debug publishers -> publisher to debug processing results in RVIZ topics
    ros::Publisher observation_pub_;
    ros::Publisher observation_BL_pub_;
    ros::Publisher map_element_pub_;
    ros::Publisher line_marker_pub;


};

DataFusion::DataFusion(const std::string& configFilePath, const std::string& mapFilePath)
: mapFilePath(mapFilePath)
{
    /*
    DataFusion Class constructor 
    */
    ros::NodeHandle nh;
    observation_pub_ = nh.advertise<geometry_msgs::PoseArray>("observation", 10);
    observation_BL_pub_ = nh.advertise<geometry_msgs::PoseArray>("observation_BL", 10);
    map_element_pub_ = nh.advertise<geometry_msgs::PoseArray>("map_element", 10, true);
    line_marker_pub = nh.advertise<visualization_msgs::Marker>("mahalanobis_distance", 10);

    // Read configuration file
    readConfig(configFilePath);

    // Read map
    map = loadMapFromCSV();
    //DEBUG >> Publish map elements
    publishPoseElements(map, map_element_pub_);

    int mapSize = map.size();
    ROS_INFO("EKF Loaded %d map elements", mapSize);

    // Initialize frame to 0 in constructor
    frame_n = 0;

    /* EKF initialization*/
    // Initial pose
    kalmanPose.x = config_.x_init;
    kalmanPose.y = config_.y_init;
    kalmanPose.z = config_.z_init;
    kalmanPose.roll = config_.roll_init;
    kalmanPose.pitch = config_.pitch_init;
    kalmanPose.yaw = config_.yaw_init;


    P = P.Zero();
    P(0,0) = config_.P00_init;
    P(1,1) = config_.P11_init;
    P(2,2) = config_.P22_init;
    P(3,3) = config_.P33_init;
    P(4,4) = config_.P44_init;
    P(5,5) = config_.P55_init;
    PFactor = config_.PFactor;
    P = P*PFactor;

    // std::cout << "Initial P: " << std::endl;
    // std::cout << P << std::endl;

    sigma_obs.x = config_.sigma_obs_x;
    sigma_obs.y = config_.sigma_obs_y;
    sigma_obs.z = config_.sigma_obs_z;
    sigma_obs.roll = config_.sigma_obs_roll;
    sigma_obs.pitch = config_.sigma_obs_pitch;
    sigma_obs.yaw = config_.sigma_obs_yaw;
    R = R.Zero();
    R(0, 0) = std::pow(sigma_obs.x, 2);
    R(1, 1) = std::pow(sigma_obs.y, 2);
    R(2, 2) = std::pow(sigma_obs.z, 2);
    R(3, 3) = std::pow(sigma_obs.roll, 2);
    R(4, 4) = std::pow(sigma_obs.pitch, 2);
    R(5, 5) = std::pow(sigma_obs.yaw, 2);

    // std::cout << "Initial R: " << std::endl;
    // std::cout << R << std::endl;


    sigma_odom.x = config_.sigma_odom_x;
    sigma_odom.y = config_.sigma_odom_y;
    sigma_odom.z = config_.sigma_odom_z;
    sigma_odom.roll = config_.sigma_odom_roll;
    sigma_odom.pitch = config_.sigma_odom_pitch;
    sigma_odom.yaw = config_.sigma_odom_yaw;
    Q = Q.Zero();
    Q(0, 0) = std::pow(sigma_odom.x, 2);
    Q(1, 1) = std::pow(sigma_odom.y, 2);
    Q(2, 2) = std::pow(sigma_odom.z, 2);
    Q(3, 3) = std::pow(sigma_odom.roll, 2);
    Q(4, 4) = std::pow(sigma_odom.pitch, 2);
    Q(5, 5) = std::pow(sigma_odom.yaw, 2);

    QFactor = config_.QFactor;
    Q = Q*QFactor;

    // std::cout << "Initial Q: " << std::endl;
    // std::cout << Q << std::endl;
}

void DataFusion::readConfig(const std::string &filename)
{
    namespace po = boost::program_options;
    po::options_description config("Configuration");
    config.add_options()
        ("data_fusion.x_init", po::value<float>(&config_.x_init)->default_value(0.0), "Initial Global X")
        ("data_fusion.y_init", po::value<float>(&config_.y_init)->default_value(0.0), "Initial Global Y")
        ("data_fusion.z_init", po::value<float>(&config_.z_init)->default_value(0.0), "Initial Global Z")
        ("data_fusion.roll_init", po::value<float>(&config_.roll_init)->default_value(0.0), "Initial Global Roll")
        ("data_fusion.pitch_init", po::value<float>(&config_.pitch_init)->default_value(0.0), "Initial Global Pitch")
        ("data_fusion.yaw_init", po::value<float>(&config_.yaw_init)->default_value(0.0), "Initial Global Yaw")
        ("data_fusion.sigma_odom_x", po::value<float>(&config_.sigma_odom_x)->default_value(0.0), "Odometry sigma X")
        ("data_fusion.sigma_odom_y", po::value<float>(&config_.sigma_odom_y)->default_value(0.0), "Odometry sigma Y")
        ("data_fusion.sigma_odom_z", po::value<float>(&config_.sigma_odom_z)->default_value(0.0), "Odometry sigma Z")
        ("data_fusion.sigma_odom_roll", po::value<float>(&config_.sigma_odom_roll)->default_value(0.0), "Odometry sigma ROLL")
        ("data_fusion.sigma_odom_pitch", po::value<float>(&config_.sigma_odom_pitch)->default_value(0.0), "Odometry sigma PITCH")
        ("data_fusion.sigma_odom_yaw", po::value<float>(&config_.sigma_odom_yaw)->default_value(0.0), "Odometry sigma YAW")
        ("data_fusion.sigma_obs_x", po::value<float>(&config_.sigma_obs_x)->default_value(0.0), "Observation sigma X")
        ("data_fusion.sigma_obs_y", po::value<float>(&config_.sigma_obs_y)->default_value(0.0), "Observation sigma Y")
        ("data_fusion.sigma_obs_z", po::value<float>(&config_.sigma_obs_z)->default_value(0.0), "Observation sigma Z")
        ("data_fusion.sigma_obs_roll", po::value<float>(&config_.sigma_obs_roll)->default_value(0.0), "Observation sigma ROLL")
        ("data_fusion.sigma_obs_pitch", po::value<float>(&config_.sigma_obs_pitch)->default_value(0.0), "Observation sigma PITCH")
        ("data_fusion.sigma_obs_yaw", po::value<float>(&config_.sigma_obs_yaw)->default_value(0.0), "Observation sigma YAW")
        ("data_fusion.mahalanobisDistanceThreshold", po::value<float>(&config_.mahalanobisDistanceThreshold)->default_value(0.0), "Mahalanobis distance threshold")
        ("data_fusion.QFactor", po::value<float>(&config_.QFactor)->default_value(0.00015), "Q Factor")
        ("data_fusion.P00_init", po::value<float>(&config_.P00_init)->default_value(0.1), "P00 init")
        ("data_fusion.P11_init", po::value<float>(&config_.P11_init)->default_value(0.1), "P11 init")
        ("data_fusion.P22_init", po::value<float>(&config_.P22_init)->default_value(1.0), "P22 init")
        ("data_fusion.P33_init", po::value<float>(&config_.P33_init)->default_value(1.0), "P33 init")
        ("data_fusion.P44_init", po::value<float>(&config_.P44_init)->default_value(1.0), "P44 init")
        ("data_fusion.P55_init", po::value<float>(&config_.P55_init)->default_value(0.1), "P55 init")
        ("data_fusion.PFactor", po::value<float>(&config_.PFactor)->default_value(0.0001), "P Factor");


    po::variables_map vm;
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    po::store(po::parse_config_file(ifs, config), vm);
    po::notify(vm);
}


std::vector<pointcloud_clustering::positionRPY> DataFusion::processPoseArray(const geometry_msgs::PoseArray& poseArray)
{
    /*
    Method to parse an array of poses and convert it to Roll-Pitch-Yaw format
    */
    std::vector<pointcloud_clustering::positionRPY> observations;

    for (const auto& pose : poseArray.poses) {
        pointcloud_clustering::positionRPY obs;
        tf::Quaternion quat;

        // Extract position
        obs.x = pose.position.x;
        obs.y = pose.position.y;
        obs.z = pose.position.z;

        // Convert quaternion to roll-pitch-yaw
        quat.setX(pose.orientation.x);
        quat.setY(pose.orientation.y);
        quat.setZ(pose.orientation.z);
        quat.setW(pose.orientation.w);

        tf::Matrix3x3 quaternionToRPY(quat);
        quaternionToRPY.getEulerYPR(obs.yaw, obs.pitch, obs.roll);

        // Add to observations vector
        observations.push_back(obs);
    }

    return observations;
}


std::vector<pointcloud_clustering::positionRPY> DataFusion::loadMapFromCSV()
{
    /*
    Load CSV map with landmark known coordinates in the scene.
    */
    std::vector<pointcloud_clustering::positionRPY> map; // To store VE positions
    pointcloud_clustering::positionRPY map_aux;
    std::ifstream inputFile(mapFilePath.c_str());
    ROS_INFO("EKF Map file: %s", mapFilePath.c_str());

    if (!inputFile.is_open()) {
        throw std::runtime_error("Cannot open map file: " + mapFilePath);
    }

    int lineNumber = 0;
    while (inputFile) {
        std::string line;
        if (!std::getline(inputFile, line)) break;

        // Skip comments
        if (line[0] == '#') continue;

        std::istringstream lineStream(line);
        std::vector<double> record;

        while (lineStream) {
            std::string value;
            if (!std::getline(lineStream, value, ',')) break;
            try {
                record.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                ROS_WARN("Invalid value found in map file at line %d", lineNumber + 1);
                continue;
            }
        }

        // Check for exactly 3 values (x, y, z)
        if (record.size() != 3) {
            ROS_WARN("Invalid data at line %d in map file", lineNumber + 1);
            continue;
        }

        // Fill map_aux with parsed data
        map_aux.x = record[0];
        map_aux.y = record[1];
        map_aux.z = record[2];
        map_aux.roll = 0.0;
        map_aux.pitch = 0.0;
        map_aux.yaw = 0.0;
        map.push_back(map_aux);

        lineNumber++;
    }

    if (!inputFile.eof()) {
        ROS_ERROR("Error reading map file");
    }

    return map;
}


void DataFusion::publishPoseElements(std::vector<pointcloud_clustering::positionRPY> pose_array, ros::Publisher publisher)
{
    ROS_DEBUG("EKF publishing elements...");
    geometry_msgs::PoseArray pose_array_msg;
    pose_array_msg.header.stamp = ros::Time::now();
    pose_array_msg.header.frame_id = "map"; // Replace with your fixed frame

    for (const auto& pose : pose_array) {
        geometry_msgs::Pose pose_msg;

        // Set position
        pose_msg.position.x = pose.x;
        pose_msg.position.y = pose.y;
        pose_msg.position.z = pose.z;

        // Convert RPY to quaternion
        tf2::Quaternion q;
        q.setRPY(pose.roll, pose.pitch, pose.yaw);

        // Set orientation
        pose_msg.orientation.x = q.x();
        pose_msg.orientation.y = q.y();
        pose_msg.orientation.z = q.z();
        pose_msg.orientation.w = q.w();

        // Add the pose to the PoseArray
        pose_array_msg.poses.push_back(pose_msg);
    }

    // Publish the PoseArray
    publisher.publish(pose_array_msg);
}



bool DataFusion::dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req, pointcloud_clustering::data_fusion_srv::Response &res)
{
    /* 
    MAIN FUNCTION
    Receives the request and generates a response
    */
   frame_n = frame_n + 1;

    /* Service Input values*/
    // Odometry input
    pointcloud_clustering::positionRPY incOdomEKF;
    // Vehicle odometry pose
    incOdomEKF.x = req.odometry.x;
    incOdomEKF.y = req.odometry.y;
    incOdomEKF.z = req.odometry.z;
    incOdomEKF.roll = req.odometry.roll;
    incOdomEKF.pitch = req.odometry.pitch;
    incOdomEKF.yaw = req.odometry.yaw;
    ROS_DEBUG("EKF Frame %d", frame_n);
    ROS_INFO("EKF Incremental odometry received: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]", 
         incOdomEKF.x, incOdomEKF.y, incOdomEKF.z, 
         incOdomEKF.roll, incOdomEKF.pitch, incOdomEKF.yaw);

    // Observations input
    geometry_msgs::PoseArray verticalElements_BL = req.verticalElements_BL; // Detected vertical elements in vehicle frame

    // Transform observations to Roll-Pitch-Yaw format
    std::vector<pointcloud_clustering::positionRPY> observations_BL = processPoseArray(verticalElements_BL);


    // Initialize B matrix
    Matrix <float, 4, 6> B; // Binding matrix for EKF
    B << 
    1, 0, 0, 0, 0, 0, // x
    0, 1, 0, 0, 0, 0, // y
    // 0, 0, 1, 0, 0, 0, // z
    0, 0, 0, 1, 0, 0, // roll
    0, 0, 0, 0, 1, 0; // pitch
    // 0, 0, 0, 0, 0, 1; // yaw

    int B_rows = B.rows();


    // Initialize Mahalanobis distance threshold for matching step
    float mahalanobisDistanceThreshold = config_.mahalanobisDistanceThreshold;

    ///////////////////////////////////////////////////////
    /* MAIN PROCESS*/
    ///////////////////////////////////////////////////////
    /* 1. Pose Prediction*/
    kalmanPose = Comp(kalmanPose, incOdomEKF);
    ROS_INFO("EKF Pose Predicted: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]", 
         kalmanPose.x, kalmanPose.y, kalmanPose.z, 
         kalmanPose.roll, kalmanPose.pitch, kalmanPose.yaw);

    // Position covariance matrix update
    Matrix6f Fx, Fu;
    Fx = J1_n(kalmanPose, incOdomEKF);
    Fu = J2_n(kalmanPose, incOdomEKF);

    // State coveriance
    P = Fx*P*Fx.transpose() + Fu*Q*Fu.transpose();

    /* 2. Get observations (Matches)*/
    int obsSize = observations_BL.size(); // Number of observations
    ROS_INFO("EKF Number of observations received: %d", obsSize);

    bool matched = false;

    if (observations_BL.empty()) {
        ROS_DEBUG("EKF No observations received. Skipping update.");
        // std::cout << "P:" << std::endl;
        // std::cout << P << std::endl;

        // Set the response
        res.corrected_position = kalmanPose;
    }
    else {
        ROS_DEBUG("EKF OBSERVATIONS FOUND >> MATCHING....");
        std::vector<int> i_vec; // Indices of matched observations
        std::vector<int> j_vec; // Indices of matched map elements

        // Data Fusion: Match observations with map elements
        std::vector<pointcloud_clustering::positionRPY> observations_map;
        
        for (int i = 0; i < observations_BL.size(); i++) {
            float minMahalanobis = mahalanobisDistanceThreshold;
            int bestMatchIndex = -1;
            //TODO: Remove >> Check distances computed
            std::vector<float> distances;

            // Project observation to map frame
            pointcloud_clustering::positionRPY observation_origin = transformPose(observations_BL[i], kalmanPose);
            observations_map.push_back(observation_origin);

            for (int j = 0; j < map.size(); j++) {
                // Compute innovation vector
                auto h_ij = computeInnovation(observation_origin, map[j], B);

                // Compute Jacobians
                auto H_z_ij = B * J2_n(Inv(map[j]), observation_origin) * J2_n(kalmanPose, observations_BL[i]);
                auto H_x_ij = B * J2_n(Inv(map[j]), observation_origin) * J1_n(kalmanPose, observations_BL[i]);

                // Innovation covariance
                auto S_ij = H_x_ij * P * H_x_ij.transpose() + H_z_ij * R * H_z_ij.transpose();

                // Compute Mahalanobis distance
                float distance = sqrt(mahalanobisDistance(h_ij, S_ij));

                //TODO: Remove
                distances.push_back(distance);
                if (distance < minMahalanobis) {
                    // Match found
                    ROS_INFO("EKF MATCH FOUND in observation %d with map element %d. Distance = %4f", i, j, distance);
                    // std::cout << "Distance: " << distance << std::endl;
                    minMahalanobis = distance;
                    bestMatchIndex = j;
                }
            }
            // Register match if found
            if (bestMatchIndex != -1) {
                matched = true;
                i_vec.push_back(i);
                j_vec.push_back(bestMatchIndex);
            }
            //TODO: Remove - Element with the minimum distance between them
            if (!distances.empty()) {
                float minDistance = *std::min_element(distances.begin(), distances.end());
                int minDistanceIndex = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
                // std::cout << "Map Element: " << map[minDistanceIndex] << std::endl;
                std::cout << "Frame: " << frame_n << " - Minimum Mahalanobis distance for Observation " << i << " and Map Element " << minDistanceIndex << " : " << minDistance << std::endl;

                // Draw line between observation and map point and publish it to ROS
                visualization_msgs::Marker line_marker;
                line_marker.header.frame_id = "map"; // Adjust the frame id as needed
                line_marker.header.stamp = ros::Time::now();
                line_marker.ns = "mahalanobis_lines";
                line_marker.id = i; // Unique ID for the marker
                line_marker.type = visualization_msgs::Marker::LINE_STRIP;
                line_marker.action = visualization_msgs::Marker::ADD;
                geometry_msgs::Point p1, p2;
                // Map point
                p1.x = map[minDistanceIndex].x;
                p1.y = map[minDistanceIndex].y;
                p1.z = map[minDistanceIndex].z;
                // Observation point in map frame
                p2.x = observation_origin.x;
                p2.y = observation_origin.y;
                p2.z = observation_origin.z;
                line_marker.points.push_back(p1);
                line_marker.points.push_back(p2);
                // Set line properties
                line_marker.scale.x = 0.05; // Line width
                // Line color (RGBA)
                line_marker.color.r = 1.0;
                line_marker.color.g = 0.0;
                line_marker.color.b = 0.0;
                line_marker.color.a = 1.0;
                // Publish the marker
                line_marker_pub.publish(line_marker);

                // Add text for the distance
                visualization_msgs::Marker text;
                text.header.frame_id = "map";
                text.header.stamp = ros::Time::now();
                text.ns = "distance_text";
                text.id = 1; // Set a unique ID
                text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                text.action = visualization_msgs::Marker::ADD;
                text.pose.position.x = (p1.x + p2.x) / 2.0;
                text.pose.position.y = (p1.y + p2.y) / 2.0;
                text.pose.position.z = (p1.z + p2.z) / 2.0;
                text.scale.z = 0.5;
                text.text = std::to_string(minDistance);
                text.color.r = 1.0;
                text.color.g = 1.0;
                text.color.b = 1.0;
                text.color.a = 1.0;
                text.scale.z = 0.2;
                line_marker_pub.publish(text);
            }
        }

        // DEBUG: Publish observations in map frame
        publishPoseElements(observations_map, observation_pub_);
        publishPoseElements(observations_BL, observation_BL_pub_);

        // If no matches were found, skip update
        if (!matched) {
            ROS_DEBUG("EKF No matches found. Using predicted state.");
            // Set the response
            res.corrected_position = kalmanPose;
        }
        else {
            //TODO: Refactorizar, se repiten fórmulas que ya se han hecho anteriormente
            // Initialize matrices for correction
            int M = i_vec.size(); // Number of matches
            ROS_DEBUG("EKF Matches found: %d", M);
            MatrixXf h_k(M * B.rows(), 1);
            MatrixXf H_x_k(M * B.rows(), 6);
            MatrixXf H_z_k(M * B.rows(), M * 6);
            MatrixXf R_k(M * 6, M * 6);

            h_k.setZero();
            H_x_k.setZero();
            H_z_k.setZero();
            R_k.setZero();

            // Populate matrices based on matches
            for (int m = 0; m < M; m++) {
                int i = i_vec[m];
                int j = j_vec[m];
                // Project observation to map frame
                pointcloud_clustering::positionRPY observation_origin = Comp(kalmanPose, observations_BL[i]);
                // Compute innovation
                auto h_i = computeInnovation(observation_origin, map[j], B);
                h_k.block(m * B.rows(), 0, B.rows(), 1) = h_i;

                // Compute Jacobians
                auto H_x_i = B * J2_n(Inv(map[j]), observation_origin) * J1_n(kalmanPose, observations_BL[i]);
                H_x_k.block(m * B.rows(), 0, B.rows(), 6) = H_x_i;

                auto H_z_i = B * J2_n(Inv(map[j]), observation_origin) * J2_n(kalmanPose, observations_BL[i]);
                
                H_z_k.block(m * B.rows(), m * 6, B.rows(), 6) = H_z_i;
                // Add observation noise
                R_k.block(m * 6, m * 6, 6, 6) = R;
            }

            // Compute innovation covariance
            auto S_k = H_x_k * P * H_x_k.transpose() + H_z_k * R_k * H_z_k.transpose();

            // Compute Kalman gain
            auto W = P * H_x_k.transpose() * S_k.inverse();

            // Update state
            kalmanPose = vec2RPY(RPY2Vec(kalmanPose) - W * h_k);

            // Update covariance
            Matrix6f updatedP = (Matrix6f::Identity() - W * H_x_k) * P;

            ROS_INFO("EKF Corrected Position: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                    kalmanPose.x, kalmanPose.y, kalmanPose.z,
                    kalmanPose.roll, kalmanPose.pitch, kalmanPose.yaw);

            P = updatedP;

            // Set the response
            res.corrected_position = kalmanPose;
        }
    }

    // Clear received input
    // req.verticalElements_BL.poses.clear();
    verticalElements_BL.poses.clear();
    observations_BL.clear();
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "data_fusion_service");
    ros::NodeHandle nh;

    // Current working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        ROS_DEBUG("Current working directory: %s", cwd);
    } else {
        ROS_ERROR("Error getting current working directory");
    }

    // Get config path parameter
    std::string configFilePath;
    if (!nh.getParam("data_fusion_config_file_path", configFilePath)) {
        ROS_ERROR("Failed to get param 'data_fusion_config_file_path'");
        return -1;
    }
    // Get map path parameter
    std::string mapFilePath;
    if (!nh.getParam("data_fusion_map_file_path", mapFilePath)) {
        ROS_ERROR("Failed to get param 'data_fusion_map_file_path'");
        return -1;
    }
    try {
        DataFusion df(configFilePath, mapFilePath);
        ros::ServiceServer service = nh.advertiseService("data_fusion", &DataFusion::dataFusionService, &df);
        ROS_DEBUG("Service Data Fusion initialized correctly");

        ros::spin();
    } catch (const std::exception &e) {
        ROS_ERROR("Error initializing DataFusionService: %s", e.what());
        return -1;
    }


    return 0;
}
