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

struct Config {
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
};

class DataFusion {
public:
    DataFusion(const std::string& configFilePath, const std::string& mapFilePath);
    bool dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req,
                                  pointcloud_clustering::data_fusion_srv::Response &res);
private:
    void readConfig(const std::string &filename);

    Config config_;
    std::string frame_id_;
    Matrix6f P_;
    Matrix6f Q_;
    Matrix6f R_;
    std::vector<pointcloud_clustering::observationRPY> map_;
};

DataFusion::DataFusion(const std::string& configFilePath, const std::string& mapFilePath) {
    ros::NodeHandle nh;
    readConfig(configFilePath);

    // Inicializa las matrices P, Q y R con valores por defecto
    P_ = Matrix6f::Zero();
    Q_ = Matrix6f::Zero();
    R_ = Matrix6f::Zero();

    // Inicialización de las matrices Q y R usando la configuración
    Q_(0, 0) = config_.sigma_odom_x * config_.sigma_odom_x;
    Q_(1, 1) = config_.sigma_odom_y * config_.sigma_odom_y;
    Q_(2, 2) = config_.sigma_odom_z * config_.sigma_odom_z;
    Q_(3, 3) = config_.sigma_odom_roll * config_.sigma_odom_roll;
    Q_(4, 4) = config_.sigma_odom_pitch * config_.sigma_odom_pitch;
    Q_(5, 5) = config_.sigma_odom_yaw * config_.sigma_odom_yaw;

    R_(0, 0) = config_.sigma_obs_x * config_.sigma_obs_x;
    R_(1, 1) = config_.sigma_obs_y * config_.sigma_obs_y;
    R_(2, 2) = config_.sigma_obs_z * config_.sigma_obs_z;
    R_(3, 3) = config_.sigma_obs_roll * config_.sigma_obs_roll;
    R_(4, 4) = config_.sigma_obs_pitch * config_.sigma_obs_pitch;
    R_(5, 5) = config_.sigma_obs_yaw * config_.sigma_obs_yaw;

    // Carga del mapa desde un archivo JSON (suponiendo que el archivo JSON esté en mapFilePath)
    std::ifstream inputFile(mapFilePath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Cannot open map file: " + mapFilePath);
    }

    nlohmann::json jsonData;
    inputFile >> jsonData;

    for (const auto& item : jsonData) {
        if (item.contains("stopSign")) {
            auto stopSign = item["stopSign"];
            pointcloud_clustering::observationRPY map_aux;
            map_aux.position.x = stopSign["position"]["x"].get<double>() - config_.easting_ref;
            map_aux.position.y = stopSign["position"]["y"].get<double>() - config_.northing_ref;
            map_aux.position.z = stopSign["position"]["z"].get<double>();  // Z coordinate from the JSON
            map_aux.position.roll = 0.0;
            map_aux.position.pitch = 0.0;
            map_aux.position.yaw = 0.0;
            map_.push_back(map_aux);
        }
    }
    inputFile.close();


    // // Carga del mapa desde un archivo CSV (suponiendo que el archivo CSV esté en mapFilePath)
    // std::ifstream inputFile(mapFilePath);
    // if (!inputFile.is_open()) {
    //     throw std::runtime_error("Cannot open map file: " + mapFilePath);
    // }

    // while (inputFile) {
    //     std::string s;
    //     if (!std::getline(inputFile, s)) break;
    //     if (s[0] != '#') {
    //         std::istringstream ss(s);
    //         std::vector<double> record;

    //         while (ss) {
    //             std::string line;
    //             if (!std::getline(ss, line, ','))
    //                 break;
    //             record.push_back(std::stof(line));
    //         }
    //         pointcloud_clustering::observationRPY map_aux;
    //         map_aux.position.x = record[0] - config_.easting_ref;
    //         map_aux.position.y = record[1] - config_.northing_ref;
    //         map_aux.position.z = -1.8;  // Valor fijo
    //         map_aux.position.roll = 0.0;
    //         map_aux.position.pitch = 0.0;
    //         map_aux.position.yaw = 0.0;
    //         map_.push_back(map_aux);
    //     }
    // }
    // inputFile.close();
}

void DataFusion::readConfig(const std::string &filename) {
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
        ("data_fusion.mahalanobisDistanceThreshold", po::value<float>(&config_.mahalanobisDistanceThreshold)->default_value(0.0), "Mahalanobis distance threshold");

    po::variables_map vm;
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    po::store(po::parse_config_file(ifs, config), vm);
    po::notify(vm);
}

bool DataFusion::dataFusionService(pointcloud_clustering::data_fusion_srv::Request &req,
                                  pointcloud_clustering::data_fusion_srv::Response &res) {
    pointcloud_clustering::positionRPY positionPredEKF;
    pointcloud_clustering::positionRPY positionCorrEKF;
    Matrix6f Fx, Fu;

    // Obtener la odometría y las observaciones del request
    geometry_msgs::PoseStamped odometry = req.odometry;
    geometry_msgs::PoseArray observations = req.observations;

    // Convertir la odometría en el formato interno
    pointcloud_clustering::positionRPY incOdomEKF;
    incOdomEKF.x = odometry.pose.position.x;
    incOdomEKF.y = odometry.pose.position.y;
    incOdomEKF.z = odometry.pose.position.z;
    tf::Quaternion quat;
    tf::quaternionMsgToTF(odometry.pose.orientation, quat);
    tf::Matrix3x3(quat).getRPY(incOdomEKF.roll, incOdomEKF.pitch, incOdomEKF.yaw);

    // Predicción de la posición
    positionPredEKF = Comp(positionCorrEKF, incOdomEKF);
    
    // Actualización de la matriz de covarianza de predicción
    Fx = J1_n(positionCorrEKF, incOdomEKF);
    Fu = J2_n(positionCorrEKF, incOdomEKF);
    P_ = Fx * P_ * Fx.transpose() + Fu * Q_ * Fu.transpose();

    // Actualización utilizando las observaciones
    std::vector<pointcloud_clustering::observationRPY> obs_list;
    for (const auto& pose : observations.poses) {
        pointcloud_clustering::observationRPY obs;
        obs.position.x = pose.position.x;
        obs.position.y = pose.position.y;
        obs.position.z = pose.position.z;
        tf::Quaternion obs_quat;
        tf::quaternionMsgToTF(pose.orientation, obs_quat);
        tf::Matrix3x3(obs_quat).getRPY(obs.position.roll, obs.position.pitch, obs.position.yaw);
        obs_list.push_back(obs);
    }

    int M = 0;
    std::vector<int> i_vec, j_vec;

    for (size_t i = 0; i < obs_list.size(); ++i) {
        for (size_t j = 0; j < map_.size(); ++j) {
            auto h_ij = RPY2Vec(Comp(Inv(map_[j].position), Comp(positionPredEKF, obs_list[i].position)));
            auto H_x_ij = J2_n(Inv(map_[j].position), Comp(positionPredEKF, obs_list[i].position)) * J1_n(positionPredEKF, obs_list[i].position);
            auto H_z_ij = J2_n(Inv(map_[j].position), Comp(positionPredEKF, obs_list[i].position)) * J2_n(positionPredEKF, obs_list[i].position);
            Matrix6f S_ij = H_x_ij * P_ * H_x_ij.transpose() + H_z_ij * R_ * H_z_ij.transpose();
            float mahalanobis = sqrt(mahalanobisDistance(h_ij, S_ij));
            if (mahalanobis < config_.mahalanobisDistanceThreshold) {
                i_vec.push_back(i);
                j_vec.push_back(j);
                ++M;
            }
        }
    }

    if (M > 0) {
        MatrixXf h_k = MatrixXf::Zero(M * 6, 1);
        MatrixXf H_x_k = MatrixXf::Zero(M * 6, 6);
        MatrixXf H_z_k = MatrixXf::Zero(M * 6, M * 6);
        MatrixXf R_k = MatrixXf::Zero(M * 6, M * 6);

        for (int m = 0; m < M; ++m) {
            auto h_i = RPY2Vec(Comp(Inv(map_[j_vec[m]].position), Comp(positionPredEKF, obs_list[i_vec[m]].position)));
            auto H_x_i = J2_n(Inv(map_[j_vec[m]].position), Comp(positionPredEKF, obs_list[i_vec[m]].position)) * J1_n(positionPredEKF, obs_list[i_vec[m]].position);
            auto H_z_i = J2_n(Inv(map_[j_vec[m]].position), Comp(positionPredEKF, obs_list[i_vec[m]].position)) * J2_n(positionPredEKF, obs_list[i_vec[m]].position);
            h_k.block(m * 6, 0, 6, 1) = h_i;
            H_x_k.block(m * 6, 0, 6, 6) = H_x_i;
            H_z_k.block(m * 6, m * 6, 6, 6) = H_z_i;
            R_k.block(m * 6, m * 6, 6, 6) = R_;
        }

        auto S_k = H_x_k * P_ * H_x_k.transpose() + H_z_k * R_k * H_z_k.transpose();
        auto W = P_ * H_x_k.transpose() * S_k.inverse();
        positionCorrEKF = vec2RPY(RPY2Vec(positionPredEKF) - W * h_k);
        P_ = (Matrix6f::Identity() - W * H_x_k) * P_;
    } else {
        positionCorrEKF = positionPredEKF;
    }

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
        ROS_INFO("Current working directory: %s", cwd);
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
        ROS_INFO("Service Data Fusion initialized correctly");

        ros::spin();
    } catch (const std::exception &e) {
        ROS_ERROR("Error initializing DataFusionService: %s", e.what());
        return -1;
    }


    return 0;
}
