#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <pointcloud_clustering/positionRPY.h>
#include <pointcloud_clustering/observationRPY.h>

using namespace Eigen;
typedef Eigen::Matrix<float, 5, 5> Matrix5f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 5, 1> Vector5f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;


//************************************************//
//            FUNCTION DECLARATIONS               //
//************************************************//

Matrix4f createHomogMatr(pointcloud_clustering::positionRPY pos);
pointcloud_clustering::positionRPY coordRPY(Matrix4f pos);
pointcloud_clustering::positionRPY Comp(pointcloud_clustering::positionRPY pos1, pointcloud_clustering::positionRPY pos2);
pointcloud_clustering::positionRPY Inv(pointcloud_clustering::positionRPY pos);
Matrix4f createHomogMatrInv(pointcloud_clustering::positionRPY pos);
Matrix6f J1_n(pointcloud_clustering::positionRPY pos1, pointcloud_clustering::positionRPY pos2);
Matrix6f J2_n(pointcloud_clustering::positionRPY pos1, pointcloud_clustering::positionRPY pos2);
Matrix6f computeHx2(pointcloud_clustering::observationRPY obs, pointcloud_clustering::positionRPY pos);
Vector6f computeInnovation(pointcloud_clustering::positionRPY pos, pointcloud_clustering::observationRPY obs, pointcloud_clustering::observationRPY map);
Matrix6f computeHz2(pointcloud_clustering::observationRPY obs, pointcloud_clustering::positionRPY pos);
float mahalanobisDistance(const MatrixXf& h, const MatrixXf& S);
float AngRango(float ang);
pointcloud_clustering::positionRPY vec2RPY(Vector6f pos);
Vector6f RPY2Vec(pointcloud_clustering::positionRPY pos);