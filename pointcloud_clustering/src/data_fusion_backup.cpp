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

struct Config
{
    float x_init;
    float y_init;
    float z_init;
    float roll_init;
    float pitch_init;
    float yaw_init;
    float easting_ref;
    float northing_ref;
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
    std::vector<pointcloud_clustering::observationRPY> processPoseArray(const geometry_msgs::PoseArray& poseArray);
private:
    void readConfig(const std::string &filename);

    Config config_;
    std::string frame_id_;
    std::string mapFilePath;

    // Debug publishers -> publisher to debug processing results in RVIZ topics
    ros::Publisher observation_pub_;
    ros::Publisher map_element_pub_;

};

DataFusion::DataFusion(const std::string& configFilePath, const std::string& mapFilePath)
: mapFilePath(mapFilePath)
{
    /*
    DataFusion Class constructor
    */
    ros::NodeHandle nh;
    observation_pub_ = nh.advertise<pointcloud_clustering::observationRPY>("observation", 1);
    map_element_pub_ = nh.advertise<pointcloud_clustering::observationRPY>("map_element", 1);
    readConfig(configFilePath);
}

void DataFusion::readConfig(const std::string &filename)
{
    namespace po = boost::program_options;
    po::options_description config("Configuration");
    config.add_options()
        ("data_fusion.x_init", po::value<float>(&config_.x_init)->default_value(0.0), "Initial X")
        ("data_fusion.y_init", po::value<float>(&config_.y_init)->default_value(0.0), "Initial Y")
        ("data_fusion.z_init", po::value<float>(&config_.z_init)->default_value(0.0), "Initial Z")
        ("data_fusion.roll_init", po::value<float>(&config_.roll_init)->default_value(0.0), "Initial ROLL")
        ("data_fusion.pitch_init", po::value<float>(&config_.pitch_init)->default_value(0.0), "Initial PITCH")
        ("data_fusion.yaw_init", po::value<float>(&config_.yaw_init)->default_value(0.0), "Initial YAW")
        ("data_fusion.easting_ref", po::value<float>(&config_.easting_ref)->default_value(0.0), "Easting reference")
        ("data_fusion.northing_ref", po::value<float>(&config_.northing_ref)->default_value(0.0), "Northing reference")
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


std::vector<pointcloud_clustering::observationRPY> DataFusion::processPoseArray(const geometry_msgs::PoseArray& poseArray)
{
    /*
    Method to parse an array of poses and convert it to Roll-Pitch-Yaw format
    */
    std::vector<pointcloud_clustering::observationRPY> observations;

    for (const auto& pose : poseArray.poses) {
        pointcloud_clustering::observationRPY obs;
        tf::Quaternion quat;

        // Extract position
        obs.position.x = pose.position.x;
        obs.position.y = pose.position.y;
        obs.position.z = pose.position.z;

        // Convert quaternion to roll-pitch-yaw
        quat.setX(pose.orientation.x);
        quat.setY(pose.orientation.y);
        quat.setZ(pose.orientation.z);
        quat.setW(pose.orientation.w);

        tf::Matrix3x3 quaternionToRPY(quat);
        quaternionToRPY.getEulerYPR(obs.position.yaw, obs.position.pitch, obs.position.roll);

        // Add to observations vector
        observations.push_back(obs);
    }

    return observations;
}


bool DataFusion::dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req, pointcloud_clustering::data_fusion_srv::Response &res)
{
    /* 
    MAIN FUNCTION
    Receives the request and generates a response
    */

    /* Service Input values*/
    // Odometry input
    pointcloud_clustering::positionRPY incOdomEKF;
    incOdomEKF = req.odometry; // Vehicle odometry pose
    ROS_INFO("EKF Incremental odometry received: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %f]", 
         incOdomEKF.x, incOdomEKF.y, incOdomEKF.z, 
         incOdomEKF.roll, incOdomEKF.pitch, incOdomEKF.yaw, 
         incOdomEKF.stamp.toSec());


    // Observations input
    geometry_msgs::PoseArray verticalElements;
    geometry_msgs::PoseArray verticalElements_BL;
    verticalElements = req.verticalElements; // Detected vertical elements in global frame
    verticalElements_BL = req.verticalElements_BL; // Detected vertical elements in vehicle frame
    // Transform observations to Roll-Pitch-Yaw format
    std::vector<pointcloud_clustering::observationRPY> observations = processPoseArray(verticalElements);
    std::vector<pointcloud_clustering::observationRPY> observations_BL = processPoseArray(verticalElements_BL);

    /* EKF initialization*/
    // Initial pose
    geometry_msgs::PoseStamped poseZero;
    geometry_msgs::PoseWithCovarianceStamped posePredEKF;
    geometry_msgs::PoseWithCovarianceStamped poseCorrEKF;
    pointcloud_clustering::positionRPY positionZero;
    pointcloud_clustering::positionRPY positionPredEKF;
    pointcloud_clustering::positionRPY positionCorrEKF;

    ros::Time initTime = ros::Time::now();
    tf::TransformBroadcaster br;
    tf::StampedTransform transform_ekf(tf::Transform::getIdentity(), initTime, "map", "ekf"); // Initialization;
    tf::Quaternion quat;
    bool tfEKF;

    // Pose initialization
    poseZero.pose.position.x = 0.0;
    poseZero.pose.position.y = 0.0;
    poseZero.pose.position.z = 0.0;
    poseZero.pose.orientation.x = 0.0;
    poseZero.pose.orientation.y = 0.0;
    poseZero.pose.orientation.z = 0.0;
    poseZero.pose.orientation.w = 1.0;
    poseZero.header.stamp = initTime;
    poseZero.header.frame_id = "map";

    posePredEKF.pose.pose = poseZero.pose; // posePredEKF init
    posePredEKF.header = poseZero.header;
    poseCorrEKF.pose.pose = poseZero.pose; // poseCorrEKF init
    poseCorrEKF.header = poseZero.header;

    positionZero.x = 0.0;
    positionZero.y = 0.0;
    positionZero.z = 0.0;
    positionZero.roll = 0.0;
    positionZero.pitch = 0.0;
    positionZero.yaw = 0.0;

    positionPredEKF = positionZero; // positionPredEKF init
    positionCorrEKF = positionZero; // positionCorrEKF init

    // Initialize B matrix
    Matrix <float, 4, 6> B; // Binding matrix for EKF
    B << 1, 0, 0, 0, 0, 0, // x
    0, 1, 0, 0, 0, 0, // y
    0, 0, 0, 1, 0, 0, // roll
    0, 0, 0, 0, 1, 0; // pitch

    int B_rows = B.rows();

    // Initialize P, R and Q matrix
    Matrix6f P = P.Zero(); // Kalman covariance matrix
    float PFactor;
    P(0,0) = config_.P00_init;
    P(1,1) = config_.P11_init;
    P(2,2) = config_.P22_init;
    P(3,3) = config_.P33_init;
    P(4,4) = config_.P44_init;
    P(5,5) = config_.P55_init;
    PFactor = config_.PFactor;
    P = P*PFactor;

    Matrix6f R = R.Zero(); // Observation covariance matrix -> sigma_obs
    pointcloud_clustering::positionRPY sigma_obs;
    sigma_obs.x = config_.sigma_obs_x;
    sigma_obs.y = config_.sigma_obs_y;
    sigma_obs.z = config_.sigma_obs_z;
    sigma_obs.roll = config_.sigma_obs_roll;
    sigma_obs.pitch = config_.sigma_obs_pitch;
    sigma_obs.yaw = config_.sigma_obs_yaw;
    R(0, 0) = sigma_obs.x*sigma_obs.x;
    R(1, 1) = sigma_obs.y*sigma_obs.y;
    R(2, 2) = sigma_obs.z*sigma_obs.z;
    R(3, 3) = sigma_obs.roll*sigma_obs.roll;
    R(4, 4) = sigma_obs.pitch*sigma_obs.pitch;
    R(5, 5) = sigma_obs.yaw*sigma_obs.yaw;

    Matrix6f Q = Q.Zero(); // Odometry covariance matrix -> sigma_odom
    pointcloud_clustering::positionRPY sigma_odom;

    sigma_odom.x = config_.sigma_odom_x;
    sigma_odom.y = config_.sigma_odom_y;
    sigma_odom.z = config_.sigma_odom_z;
    sigma_odom.roll = config_.sigma_odom_roll;
    sigma_odom.pitch = config_.sigma_odom_pitch;
    sigma_odom.yaw = config_.sigma_odom_yaw;
    Q(0, 0) = sigma_odom.x*sigma_odom.x;
    Q(1, 1) = sigma_odom.y*sigma_odom.y;
    Q(2, 2) = sigma_odom.z*sigma_odom.z;
    Q(3, 3) = sigma_odom.roll*sigma_odom.roll;
    Q(4, 4) = sigma_odom.pitch*sigma_odom.pitch;
    Q(5, 5) = sigma_odom.yaw*sigma_odom.yaw;

    float QFactor;
    QFactor = config_.QFactor;
    Q = Q*QFactor;

    // Initialize Mahalanobis distance threshold for matching step
    float mahalanobisDistanceThreshold;
    mahalanobisDistanceThreshold = config_.mahalanobisDistanceThreshold;


    /*
    Load CSV map with landmark known coordinates in the scene
    */
    std::vector<pointcloud_clustering::observationRPY> map; // fill with VE positions in document
    geometry_msgs::PoseArray map_elements;
    geometry_msgs::PoseStamped map_element;
    pointcloud_clustering::observationRPY map_aux;
    visualization_msgs::Marker map_id;
    std::ifstream inputFile(mapFilePath.c_str());
    if (!inputFile.is_open())
    {
        throw std::runtime_error("Cannot open map file: " + mapFilePath);
    }

    int l = 0;
    while (inputFile) {
        std::string s;
        if (!std::getline(inputFile, s)) break;
        if (s[0] != '#') {
            std::istringstream ss(s);
            std::vector<double> record;

            while (ss) {
                std::string line;
                if (!std::getline(ss, line, ','))
                break;
                try {
                record.push_back(std::stof(line));
                }
                catch (const std::invalid_argument e) {
                // std::cout << "NaN found in file " << " line " << l+1 << std::endl;
                e.what();
                }
            }
            map_aux.position.x = record[0];
            map_aux.position.y = record[1];
            map_aux.position.z = record[2];
            map_aux.position.roll = 0.0;
            map_aux.position.pitch = 0.0;
            map_aux.position.yaw = 0.0;
            map.push_back(map_aux);
        }
        l++;
    }
    if (!inputFile.eof()) {
        std::cerr << "Could not read file " << "\n";
    }
    int mapSize = map.size();
    ROS_DEBUG("Loaded %d map elements", mapSize);

    ///////////////////////////////////////////////////////
    /* MAIN PROCESS*/
    ///////////////////////////////////////////////////////

    /* 1. Pose Prediction*/
    positionPredEKF = Comp(positionCorrEKF, incOdomEKF);
    ROS_DEBUG("EKF Pose Predicted: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]", 
         positionPredEKF.x, positionPredEKF.y, positionPredEKF.z, 
         positionPredEKF.roll, positionPredEKF.pitch, positionPredEKF.yaw);
    
    // Position covariance matrix update
    Matrix6f Fx, Fu;
    Fx = J1_n(positionCorrEKF, positionPredEKF);
    Fu = J2_n(positionCorrEKF, positionPredEKF);

    // State coveriance
    P = Fx*P*Fx.transpose() + Fu*Q*Fu.transpose();

    /* 2. Update with observations*/
    

    int obsSize = observations.size(); // Number of observations
    ROS_DEBUG("Number of observations received: %d", obsSize);


    int M; // Number of matches Observations - Vertical Elements in map

    // Initialize matrices for EKF observation model and data association
    // - h_ij: Vector to store innovation (residual) values for all observation-map pairs.
    //         Dimensions: (B_rows * obsSize * mapSize) x 1
    //         Where:
    //           B_rows - Number of components of interest (e.g., x, y, yaw)
    //           obsSize - Number of observations (sensor data)
    //           mapSize - Number of elements in the map
    // - H_x_ij: Matrix to store partial derivatives of the observation model with respect to the state vector.
    //           Dimensions: B_rows x 6
    //           Where:
    //             6 - Dimension of the state vector (e.g., x, y, z, roll, pitch, yaw)
    // - H_z_ij: Matrix to store partial derivatives of the observation model with respect to the observation vector.
    //           Dimensions: B_rows x 6
    // - S_ij: Matrix to represent the covariance of the innovation (residual) for a specific observation-map pair.
    //         Dimensions: B_rows x B_rows
    MatrixXf h_ij(B_rows*obsSize*mapSize, 1);  h_ij = h_ij.Zero(obsSize*mapSize, 1);
    MatrixXf H_x_ij(B_rows, 6);
    H_x_ij = H_x_ij.Zero(B_rows, 6);
    MatrixXf H_z_ij(B_rows, 6);
    H_z_ij = H_z_ij.Zero(B_rows, 6);
    MatrixXf S_ij(B_rows, B_rows);
    S_ij = S_ij.Zero(B_rows, B_rows);

    if(obsSize > 0)
    {
        // If observations are received, compare them with the elements in the map
        bool match = false;
        int i_min = -1;
        int j_min = -1;
        std::vector<int> i_vec;
        std::vector<int> j_vec;

        float minMahalanobis = mahalanobisDistanceThreshold;

        // Compare all observations with all the elements of the map. If mahalanobisDistance < mahalanobisDistanceThreshold between i-th observation and j-th element, there is a match
        //TODO: find a more efficent way of doing Match search
        for (int i=0; i<obsSize; i++)
        {
            // Loop over observations
            for (int j=0; j<mapSize; j++)
            {
                // For each observation, loop over map elements
                // Innovation vector > difference beteewn observation and map element
                h_ij = B*RPY2Vec(Comp(Inv(map[j].position), Comp(positionPredEKF, observations_BL[i].position)));

                // Jacobians
                //TODO: Changed observations to global frame to match map coordinates > review
                H_x_ij = B*J2_n(Inv(map[j].position), Comp(positionPredEKF, observations_BL[i].position))*J1_n(positionPredEKF, observations_BL[i].position);
                H_z_ij = B*J2_n(Inv(map[j].position), Comp(positionPredEKF, observations_BL[i].position))*J2_n(positionPredEKF, observations_BL[i].position);

                // Innovation covariance
                S_ij = H_x_ij*P*H_x_ij.transpose() + H_z_ij*R*H_z_ij.transpose();

                // MATCH > If the Mahalanobis distance is below a predefined threshold (mahalanobisDistanceThreshold)
                // and smaller than the current minimum distance (minMahalanobis), the observation and map element are marked as a potential match.
                if(sqrt(mahalanobisDistance(h_ij, S_ij)) < mahalanobisDistanceThreshold && sqrt(mahalanobisDistance(h_ij, S_ij)) < minMahalanobis)
                {
                    match = true;
                    i_min = i;
                    j_min = j;
                    minMahalanobis = sqrt(mahalanobisDistance(h_ij, S_ij));
                }
            }
            if (match)
            {
                ROS_DEBUG("Observation %d matched with Map Element %d. Min Mahalanobis Distance: %f", i_min, j_min, minMahalanobis);
                // Add match indices to the vectors
                i_vec.push_back(i_min);
                j_vec.push_back(j_min);

                match = false; // Reset match flag after all map elements are parsed for 1 observation
            }
            minMahalanobis = mahalanobisDistanceThreshold;
        }
        // Number of matches for the frame (Observations -- Map)
        M = i_vec.size();

        if(M > 0)
        {
            // At least one valid match exists
            ROS_DEBUG("%d Matches found in the frame, proceed with the update step", M);
            for(int i=0; i<i_vec.size(); i++)
                ROS_DEBUG("Match i: %d", i_vec[i]);
            ROS_DEBUG(" ");
            for(int i=0; i<j_vec.size(); i++)
                ROS_DEBUG("Match j: %d", j_vec[i]);
            ROS_DEBUG(" ");

            // Initialize matrices for correction
            MatrixXf h_i(B_rows, 1);            h_i = h_i.Zero(B_rows, 1);   // -------> h_ij for a valid association between observation_i and map_j
            MatrixXf h_k(M*B_rows, 1);          h_k = h_k.Zero(M*B_rows, 1); // -------> All vectors h_i stacked, corresponding to valid matches between an observed element and an element in the map
            MatrixXf H_x_i(B_rows, 6);          H_x_i = H_x_i.Zero(B_rows, 6);        // -------> H_ij for a valid association
            MatrixXf H_x_k(M*B_rows, 6);        H_x_k = H_x_k.Zero(M*B_rows, 6);
            MatrixXf H_z_i(B_rows, 6);          H_z_i = H_z_i.Zero(B_rows, 6);
            MatrixXf H_z_k(M*B_rows, M*6);      H_z_k = H_z_k.Zero(M*B_rows, M*6);
            MatrixXf R_k(M*6, M*6);             R_k = R_k.Zero(M*6, M*6);
            MatrixXf S_k(M*B_rows, M*B_rows);   S_k = S_k.Zero(M*B_rows, M*B_rows);
            MatrixXf W(6, M*B_rows);            W = W.Zero(6, M*B_rows);

            /* 3. UPDATE*/
            // Loop ovear each match and update matrices
            for(int i=0; i<M; i++)
            {
                // Update innovation vector
                h_i = B*RPY2Vec(Comp(Inv(map[j_vec[i]].position), Comp(positionPredEKF, observations_BL[i_vec[i]].position))); // ----> Observations as seen from base_link
                h_k.block(i*B_rows, 0, B_rows, 1) = h_i;
                // Compute the Jacobian of the observation model with respect to the state
                H_x_i = B*J2_n(Inv(map[j_vec[i]].position), Comp(positionPredEKF, observations_BL[i_vec[i]].position))*J1_n(positionPredEKF, observations_BL[i_vec[i]].position);
                H_x_k.block(i*B_rows, 0, B_rows, 6) = H_x_i;
                // Compute the Jacobian of the observation model with respect to the observation
                H_z_i = B*J2_n(Inv(map[j_vec[i]].position), Comp(positionPredEKF, observations_BL[i_vec[i]].position))*J2_n(positionPredEKF, observations_BL[i_vec[i]].position);
                H_z_k.block(i*B_rows, i*6, B_rows, 6) = H_z_i;
                // Add observation noise covariance for this match
                R_k.block(i*6, i*6, 6, 6) = R;
            }
            // Compute innovation covarizance
            S_k = H_x_k*P*H_x_k.transpose() + H_z_k*R_k*H_z_k.transpose();
            // Compute Kalman Gain
            W = P*H_x_k.transpose()*S_k.inverse();

            // UPDATE State estimation
            positionCorrEKF = vec2RPY(RPY2Vec(positionPredEKF) - W*h_k);
            P = (Matrix6f::Identity() - W*H_x_k)*P; // State covariance

            ROS_DEBUG("EKF Corrected Position >> Vertical Elements and matces are found: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
             positionCorrEKF.x, positionCorrEKF.y, positionCorrEKF.z, positionCorrEKF.roll, positionCorrEKF.pitch, positionCorrEKF.yaw);
        }
        else
        {
            // vertical elements found but no matches
            positionCorrEKF = positionPredEKF;
            ROS_DEBUG("EKF Corrected Position >> Vertical elements found but No matches: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
             positionCorrEKF.x, positionCorrEKF.y, positionCorrEKF.z, positionCorrEKF.roll, positionCorrEKF.pitch, positionCorrEKF.yaw);
        }
    }
    else
    {
        // no vertical elements found
        positionCorrEKF = positionPredEKF;
        ROS_DEBUG("EKF Corrected Position >> No vertical elements found: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                positionCorrEKF.x, positionCorrEKF.y, positionCorrEKF.z, positionCorrEKF.roll, positionCorrEKF.pitch, positionCorrEKF.yaw);
    }

    
    //TODO: This code is only used to represent the corrected pose in a ROS message with covariance >> This information should be returned by the service
    // Populate the corrected position to a corrected pose in a ROS message format
    // The ROS message has the covariance of each element of the corrected pose to show the uncertainity of the prediction
    poseCorrEKF.pose.pose.position.x = positionCorrEKF.x;
    poseCorrEKF.pose.pose.position.y = positionCorrEKF.y;
    poseCorrEKF.pose.pose.position.z = 0.0; //-------------> Ignored
    for(int i=0; i<6; i++)
    {
        for(int j=0; j<6; j++)
            poseCorrEKF.pose.covariance[i+j] = P(i, j);
    }
    poseCorrEKF.header.frame_id = "map";
    poseCorrEKF.header.stamp = ros::Time::now();

    /*
    Correct orientation of the predicted pose by the EKF
    */
   // Determine the robot's corrected orientation based on the number of matches (M)
    // If fewer than 2 matches are found, rely on odometry for yaw correction (positionCorrEKF.yaw)
    // If 2 or more matches exist, use the predicted yaw (positionPredEKF.yaw) for orientation correction
    tf::Quaternion quat_msg;
    if (M < 2) // If there are two or more matches, rotate to correct yaw; otherwise, correct only (x,y) position according to odometry 
        quat_msg.setRPY(0.0, 0.0, -positionCorrEKF.yaw);
    else
        quat_msg.setRPY(0.0, 0.0, -positionPredEKF.yaw);
    
    // Convert the corrected quaternion to a ROS-compatible format and assign it to poseCorrEKF
    // The quaternion represents the robot's corrected orientation
    tf::quaternionTFToMsg(quat_msg, poseCorrEKF.pose.pose.orientation); // set quaternion in msg from tf::Quaternion
    ROS_DEBUG("PoseWithCovarianceStamped: position: [x: %f, y: %f, z: %f], orientation: [x: %f, y: %f, z: %f, w: %f]",
          poseCorrEKF.pose.pose.position.x,
          poseCorrEKF.pose.pose.position.y,
          poseCorrEKF.pose.pose.position.z,
          poseCorrEKF.pose.pose.orientation.x,
          poseCorrEKF.pose.pose.orientation.y,
          poseCorrEKF.pose.pose.orientation.z,
          poseCorrEKF.pose.pose.orientation.w);

    // ROS_DEBUG("Covariance matrix:");
    // for (int i = 0; i < 6; i++) {
    //     ROS_DEBUG("[%f, %f, %f, %f, %f, %f]",
    //             poseCorrEKF.pose.covariance[i * 6 + 0],
    //             poseCorrEKF.pose.covariance[i * 6 + 1],
    //             poseCorrEKF.pose.covariance[i * 6 + 2],
    //             poseCorrEKF.pose.covariance[i * 6 + 3],
    //             poseCorrEKF.pose.covariance[i * 6 + 4],
    //             poseCorrEKF.pose.covariance[i * 6 + 5]);
    // }

    transform_ekf.setOrigin(tf::Vector3(poseCorrEKF.pose.pose.position.x, poseCorrEKF.pose.pose.position.y, poseCorrEKF.pose.pose.position.z));
    transform_ekf.setRotation(quat_msg);

    br.sendTransform(tf::StampedTransform(transform_ekf, ros::Time::now(), "map", "ekf"));

    if(tfEKF)
        br.sendTransform(tf::StampedTransform(tf::Transform::getIdentity(), ros::Time::now(), "ekf", "base_link")); // -------> Plug /base_link to /ekf frame

    // Preparar la respuesta
    res.corrected_position = positionCorrEKF;

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
