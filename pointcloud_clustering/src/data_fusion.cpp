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
    bool dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req,
                                  pointcloud_clustering::data_fusion_srv::Response &res);
private:
    void readConfig(const std::string &filename);

    Config config_;
    std::string frame_id_;
    std::string mapFilePath;

};

DataFusion::DataFusion(const std::string& configFilePath, const std::string& mapFilePath)
: mapFilePath(mapFilePath)
{
    /*
    Class constructor
    */
    ros::NodeHandle nh;
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

bool DataFusion::dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req, pointcloud_clustering::data_fusion_srv::Response &res)
{
    /* 
    MAIN FUNCTION
    Receives the request and generates a response
    */

    /* EKF INITIALIZATION*/
    // Input values
    pointcloud_clustering::positionRPY incOdomEKF;
    geometry_msgs::PoseArray verticalElements;
    geometry_msgs::PoseArray verticalElements_BL;
    pointcloud_clustering::positionRPY incPositionOdom;
    // Receive input values
    incOdomEKF = req.odometry; // Vehicle odometry pose
    ROS_INFO("Incremental odometry: x=%.2f, y=%.2f, z=%.2f, roll=%.2f, pitch=%.2f, yaw=%.2f, stamp=%f", 
         incOdomEKF.x, incOdomEKF.y, incOdomEKF.z, 
         incOdomEKF.roll, incOdomEKF.pitch, incOdomEKF.yaw, 
         incOdomEKF.stamp.toSec());
    verticalElements = req.verticalElements; // Detected vertical elements in global frame
    verticalElements_BL = req.verticalElements_BL; // Detected vertical elements in vehicle frame

    // EKF initialization
    geometry_msgs::PoseStamped poseZero;

    geometry_msgs::PoseWithCovarianceStamped posePredEKF;
    geometry_msgs::PoseWithCovarianceStamped poseCorrEKF;

    pointcloud_clustering::positionRPY positionZero;
    pointcloud_clustering::positionRPY positionEKF;
    pointcloud_clustering::positionRPY incOdomEKFPrev;  
    pointcloud_clustering::positionRPY positionPredEKF;
    pointcloud_clustering::positionRPY positionCorrEKF;
    pointcloud_clustering::positionRPY sigma_odom;
    pointcloud_clustering::positionRPY sigma_obs;

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

    positionEKF.x = config_.x_init;
    positionEKF.y = config_.y_init;
    positionEKF.z = config_.z_init;
    positionEKF.roll = config_.roll_init;
    positionEKF.pitch = config_.pitch_init;
    positionEKF.yaw = config_.yaw_init;
    positionEKF.yaw = positionEKF.yaw*3.141592/180.0;

    float easting_ref, northing_ref;
    easting_ref = config_.easting_ref;
    northing_ref = config_.northing_ref;

    sigma_odom.x = config_.sigma_odom_x;
    sigma_odom.y = config_.sigma_odom_y;
    sigma_odom.z = config_.sigma_odom_z;
    sigma_odom.roll = config_.sigma_odom_roll;
    sigma_odom.pitch = config_.sigma_odom_pitch;
    sigma_odom.yaw = config_.sigma_odom_yaw;

    sigma_obs.x = config_.sigma_obs_x;
    sigma_obs.y = config_.sigma_obs_y;
    sigma_obs.z = config_.sigma_obs_z;
    sigma_obs.roll = config_.sigma_obs_roll;
    sigma_obs.pitch = config_.sigma_obs_pitch;
    sigma_obs.yaw = config_.sigma_obs_yaw;

    Matrix <float, 4, 6> B; // Binding matrix for EKF
    B << 1, 0, 0, 0, 0, 0, // x
    0, 1, 0, 0, 0, 0, // y
    0, 0, 0, 1, 0, 0, // roll
    0, 0, 0, 0, 1, 0; // pitch

    int B_rows = B.rows();

    std::vector<pointcloud_clustering::observationRPY> map_;

    incOdomEKFPrev = positionZero; // incOdomEKFPrev init
    positionPredEKF = positionZero; // positionPredEKF init
    positionCorrEKF = positionZero; // positionCorrEKF init

    // Inicializa las matrices P, Q y R con valores por defecto
    Matrix6f P = P.Zero(); // Kalman covariance matrix
    Matrix6f R = R.Zero(); // Observation covariance matrix -> sigma_obs
    Matrix6f Q = Q.Zero(); // Odometry covariance matrix -> sigma_odom

    Matrix6f Fx, Fu;

    R(0, 0) = sigma_obs.x*sigma_obs.x;
    R(1, 1) = sigma_obs.y*sigma_obs.y;
    R(2, 2) = sigma_obs.z*sigma_obs.z;
    R(3, 3) = sigma_obs.roll*sigma_obs.roll;
    R(4, 4) = sigma_obs.pitch*sigma_obs.pitch;
    R(5, 5) = sigma_obs.yaw*sigma_obs.yaw;

    Q(0, 0) = sigma_odom.x*sigma_odom.x;
    Q(1, 1) = sigma_odom.y*sigma_odom.y;
    Q(2, 2) = sigma_odom.z*sigma_odom.z;
    Q(3, 3) = sigma_odom.roll*sigma_odom.roll;
    Q(4, 4) = sigma_odom.pitch*sigma_odom.pitch;
    Q(5, 5) = sigma_odom.yaw*sigma_odom.yaw;

    float QFactor;
    QFactor = config_.QFactor;
    Q = Q*QFactor;

    P(0,0) = config_.P00_init;
    P(1,1) = config_.P11_init;
    P(2,2) = config_.P22_init;
    P(3,3) = config_.P33_init;
    P(4,4) = config_.P44_init;
    P(5,5) = config_.P55_init;

    float PFactor;
    PFactor = config_.PFactor;
    P = P*PFactor;

    float mahalanobisDistanceThreshold;
    mahalanobisDistanceThreshold = config_.mahalanobisDistanceThreshold;

    int aux;
    double theta, thetaPrev;
    thetaPrev = config_.yaw_init;
    thetaPrev = thetaPrev*3.141592/180.0;
    theta = thetaPrev;

    // Carga del mapa desde un archivo CSV (suponiendo que el archivo CSV esté en mapFilePath)
    std::vector<pointcloud_clustering::observationRPY> map; // fill with VE positions in document
    geometry_msgs::PoseArray map_elements;
    geometry_msgs::PoseStamped map_element;
    pointcloud_clustering::observationRPY map_aux;
    std::ifstream inputFile(mapFilePath);
    if (!inputFile.is_open())
    {
        throw std::runtime_error("Cannot open map file: " + mapFilePath);
    }

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
                record.push_back(std::stof(line));
            }
            pointcloud_clustering::observationRPY map_aux;
            map_aux.position.x = record[0] - config_.easting_ref;
            map_aux.position.y = record[1] - config_.northing_ref;
            map_aux.position.z = -1.8;  // Valor fijo
            map_aux.position.roll = 0.0;
            map_aux.position.pitch = 0.0;
            map_aux.position.yaw = 0.0;
            map_.push_back(map_aux);

            map_element.pose.position.x = record[0] - easting_ref;
            map_element.pose.position.y = record[1] - northing_ref;
            map_element.pose.position.z = -1.8;
            map_element.pose.orientation.x = 0.0;
            map_element.pose.orientation.y = 0.0;
            map_element.pose.orientation.z = 0.0;
            map_element.pose.orientation.w = 1.0;
            map_elements.poses.push_back(map_element.pose);
        }
    }
    inputFile.close();
    int mapSize = map.size();
    ROS_DEBUG("Loaded %d map elements", mapSize);

    ///////////////////////////////////////////////////////
    /* MAIN PROCESS*/
    ///////////////////////////////////////////////////////
    /* 1. Predicción de la posición*/
    positionPredEKF = Comp(positionCorrEKF, incOdomEKF);
    posePredEKF.pose.pose.position.x = positionPredEKF.x;
    posePredEKF.pose.pose.position.y = positionPredEKF.y;
    posePredEKF.pose.pose.position.z = positionPredEKF.z;
    
    quat.setRPY(0.0, 0.0, -positionPredEKF.yaw);
    tf::quaternionTFToMsg(quat, posePredEKF.pose.pose.orientation);
    // std::cout << "PosePred:" << std::endl << posePredEKF.pose.pose << std::endl;
    
    // Actualización de la matriz de covarianza de predicción
    Fx = J1_n(positionCorrEKF, incOdomEKF);
    Fu = J2_n(positionCorrEKF, incOdomEKF);

    // Covarianza del estado
    P = Fx*P*Fx.transpose() + Fu*Q*Fu.transpose();

    /* 2. Actualización con las observaciones*/
    std::vector<pointcloud_clustering::observationRPY> observations;
    std::vector<pointcloud_clustering::observationRPY> observations_BL;

    for (int i=0; i<verticalElements.poses.size(); i++) // read geometry_msgs::PoseArray and store as std::vector<observationRPY>
    {
        pointcloud_clustering::observationRPY obs_aux; // ------------------------> Frame id: "map"
        tf::Quaternion quat_aux;
        obs_aux.position.x = verticalElements.poses[i].position.x;
        obs_aux.position.y = verticalElements.poses[i].position.y;
        obs_aux.position.z = verticalElements.poses[i].position.z;
        quat_aux.setX(verticalElements.poses[i].orientation.x);
        quat_aux.setY(verticalElements.poses[i].orientation.y);
        quat_aux.setZ(verticalElements.poses[i].orientation.z);
        quat_aux.setW(verticalElements.poses[i].orientation.w);
        tf::Matrix3x3 quaternionToYPR_aux(quat_aux);
        quaternionToYPR_aux.getEulerYPR(obs_aux.position.yaw, obs_aux.position.pitch, obs_aux.position.roll);
        
        pointcloud_clustering::observationRPY obs_aux_BL; // -----------------------> Frame id: "base_link"
        tf::Quaternion quat_aux_BL;
        obs_aux_BL.position.x = verticalElements_BL.poses[i].position.x;
        obs_aux_BL.position.y = verticalElements_BL.poses[i].position.y;
        obs_aux_BL.position.z = verticalElements_BL.poses[i].position.z;
        quat_aux_BL.setX(verticalElements_BL.poses[i].orientation.x);
        quat_aux_BL.setY(verticalElements_BL.poses[i].orientation.y);
        quat_aux_BL.setZ(verticalElements_BL.poses[i].orientation.z);
        quat_aux_BL.setW(verticalElements_BL.poses[i].orientation.w);
        tf::Matrix3x3 quaternionToYPR_aux_BL(quat_aux_BL);
        quaternionToYPR_aux_BL.getEulerYPR(obs_aux_BL.position.yaw, obs_aux_BL.position.pitch, obs_aux_BL.position.roll);

        observations.push_back(obs_aux);
        observations_BL.push_back(obs_aux_BL);
    }

    int obsSize = observations.size();
    int M;

    MatrixXf h_ij(B_rows*obsSize*mapSize, 1);  h_ij = h_ij.Zero(obsSize*mapSize, 1);
    MatrixXf H_x_ij(B_rows, 6);
    H_x_ij = H_x_ij.Zero(B_rows, 6);
    MatrixXf H_z_ij(B_rows, 6);
    H_z_ij = H_z_ij.Zero(B_rows, 6);
    MatrixXf S_ij(B_rows, B_rows);
    S_ij = S_ij.Zero(B_rows, B_rows);

    if(obsSize > 0)
    {
        bool match = false;
        int i_min = -1;
        int j_min = -1;
        std::vector<int> i_vec;
        std::vector<int> j_vec;

        float minMahalanobis = mahalanobisDistanceThreshold;

        for (int i=0; i<obsSize; i++) // Compare all observations with all the elements of the map. If mahalanobisDistance < mahalanobisDistanceThreshold between i-th observation and j-th element, there is a match
        {
            for (int j=0; j<mapSize; j++)
            {
                h_ij = B*RPY2Vec(Comp(Inv(map[j].position), Comp(positionPredEKF, observations_BL[i].position)));
                H_x_ij = B*J2_n(Inv(map[j].position), Comp(positionPredEKF, observations_BL[i].position))*J1_n(positionPredEKF, observations_BL[i].position);
                H_z_ij = B*J2_n(Inv(map[j].position), Comp(positionPredEKF, observations_BL[i].position))*J2_n(positionPredEKF, observations_BL[i].position);
                S_ij = H_x_ij*P*H_x_ij.transpose() + H_z_ij*R*H_z_ij.transpose();                   
                if(sqrt(mahalanobisDistance(h_ij, S_ij)) < mahalanobisDistanceThreshold && sqrt(mahalanobisDistance(h_ij, S_ij)) < minMahalanobis) //Theres is a match, but it must be the minimum value of all possible matches
                {
                    // if(match)
                    // std::cout << "***************************************REMATCH! ["<< i <<"]["<< j <<"]***************************************" << std::endl;
                    // else
                    // std::cout << "***************************************MATCH! ["<< i <<"]["<< j <<"]***************************************" << std::endl;
                    
                    match = true;
                    i_min = i;
                    j_min = j;
                    minMahalanobis = sqrt(mahalanobisDistance(h_ij, S_ij));
                }
            }
            if (match)
            {
                i_vec.push_back(i_min);
                j_vec.push_back(j_min);

                match = false;
            }
            minMahalanobis = mahalanobisDistanceThreshold;
        }
        M = i_vec.size();

        if(M > 0) // There has been at least 1 match (M=1)
        {
            // std::cout << "i_vec: ";
            // for(int i=0; i<i_vec.size(); i++)
            //     std::cout <<  i_vec[i]  << " ";
            // std::cout << std::endl;
            // std::cout << "j_vec: ";
            // for(int i=0; i<j_vec.size(); i++)
            //     std::cout <<  j_vec[i]  << " ";
            // std::cout << std::endl;

            MatrixXf h_i(B_rows, 1);            h_i = h_i.Zero(B_rows, 1);   // -------> h_ij for a valid association between observation_i and map_j
            MatrixXf h_k(M*B_rows, 1);          h_k = h_k.Zero(M*B_rows, 1); // -------> All vectors h_i stacked, corresponding to valid matches between an observed element and an element in the map
            MatrixXf H_x_i(B_rows, 6);          H_x_i = H_x_i.Zero(B_rows, 6);        // -------> H_ij for a valid association
            MatrixXf H_x_k(M*B_rows, 6);        H_x_k = H_x_k.Zero(M*B_rows, 6);
            MatrixXf H_z_i(B_rows, 6);          H_z_i = H_z_i.Zero(B_rows, 6);
            MatrixXf H_z_k(M*B_rows, M*6);      H_z_k = H_z_k.Zero(M*B_rows, M*6);
            MatrixXf R_k(M*6, M*6);             R_k = R_k.Zero(M*6, M*6);
            MatrixXf S_k(M*B_rows, M*B_rows);   S_k = S_k.Zero(M*B_rows, M*B_rows);
            MatrixXf W(6, M*B_rows);            W = W.Zero(6, M*B_rows);

            for(int i=0; i<M; i++)
            {
                h_i = B*RPY2Vec(Comp(Inv(map[j_vec[i]].position), Comp(positionPredEKF, observations_BL[i_vec[i]].position))); // ----> Observations as seen from base_link
                h_k.block(i*B_rows, 0, B_rows, 1) = h_i;
                H_x_i = B*J2_n(Inv(map[j_vec[i]].position), Comp(positionPredEKF, observations_BL[i_vec[i]].position))*J1_n(positionPredEKF, observations_BL[i_vec[i]].position);
                H_x_k.block(i*B_rows, 0, B_rows, 6) = H_x_i; 
                H_z_i = B*J2_n(Inv(map[j_vec[i]].position), Comp(positionPredEKF, observations_BL[i_vec[i]].position))*J2_n(positionPredEKF, observations_BL[i_vec[i]].position);
                H_z_k.block(i*B_rows, i*6, B_rows, 6) = H_z_i;

                R_k.block(i*6, i*6, 6, 6) = R;
            }
            S_k = H_x_k*P*H_x_k.transpose() + H_z_k*R_k*H_z_k.transpose();
            W = P*H_x_k.transpose()*S_k.inverse();
            positionCorrEKF = vec2RPY(RPY2Vec(positionPredEKF) - W*h_k);
            P = (Matrix6f::Identity() - W*H_x_k)*P;
        }
        else // vertical elements found but no matches
            positionCorrEKF = positionPredEKF;
    }
    else // no vertical elements found
        positionCorrEKF = positionPredEKF;
    
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

    tf::Quaternion quat_msg;
    if (M<2) // If there are two or more matches, rotate to correct yaw; otherwise, correct only (x,y) position according to odometry 
        quat_msg.setRPY(0.0, 0.0, -positionCorrEKF.yaw);
    else
        quat_msg.setRPY(0.0, 0.0, -positionPredEKF.yaw);
    tf::quaternionTFToMsg(quat_msg, poseCorrEKF.pose.pose.orientation); // set quaternion in msg from tf::Quaternion
    // std::cout << "poseCorrEKF: " << std::endl << poseCorrEKF.pose.pose << std::endl;

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
