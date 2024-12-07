#include <custom_functions.h>


//************************************//
//            FUNCTIONS               //
//************************************//


Vector6f RPY2Vec(pointcloud_clustering::positionRPY position){ // Conversion pointcloud_clustering::positionRPY -> Vector6f
  Vector6f result;
  
  result << position.x, position.y, position.z, position.roll, position.pitch, position.yaw;
  
  return(result);
}

/*----------------------------------------------------------------------------------------------*/
pointcloud_clustering::positionRPY Comp(pointcloud_clustering::positionRPY accumulated_pose, pointcloud_clustering::positionRPY increment) {
    Matrix4f actualMatrix = createHomogMatr(accumulated_pose);
    Matrix4f incrementMatrix = createHomogMatr(increment);

    // Multiply the matrices to accumulate the pose
    Matrix4f resultMatrix = actualMatrix * incrementMatrix;

    // Convert the result back to pose
    return(coordRPY(resultMatrix));
}
/*----------------------------------------------------------------------------------------------*/

Matrix4f createHomogMatr(pointcloud_clustering::positionRPY position){ // Homogeneous matrix (4x4) of a position given as x, y, z, roll, pitch, yaw 
    Matrix4f transform = Matrix4f::Identity();
      
    float x = position.x;
    float y = position.y;
    float z = position.z;
    float roll = position.roll;
    float pitch   = position.pitch;
    float yaw = position.yaw;

    // Rotation matrices
    Eigen::Matrix3f Rx, Ry, Rz;

    // Rotation around X-axis (roll)
    Rx << 1, 0, 0,
          0, cos(roll), -sin(roll),
          0, sin(roll), cos(roll);

    // Rotation around Y-axis (pitch)
    Ry << cos(pitch), 0, sin(pitch),
          0, 1, 0,
          -sin(pitch), 0, cos(pitch);

    // Rotation around Z-axis (yaw)
    Rz << cos(yaw), -sin(yaw), 0,
          sin(yaw), cos(yaw), 0,
          0, 0, 1;

    // Combined rotation matrix
    Eigen::Matrix3f R = Rz * Ry * Rx;

    // Assign rotation to the transformation matrix
    transform.block<3, 3>(0, 0) = R;

    // Assign translation to the transformation matrix
    transform(0, 3) = x;
    transform(1, 3) = y;
    transform(2, 3) = z;

    return transform;
}


/*----------------------------------------------------------------------------------------------*/
pointcloud_clustering::positionRPY coordRPY(Eigen::Matrix4f matrix) {
    pointcloud_clustering::positionRPY pose;
    // Extract translation
    pose.x = matrix(0, 3);
    pose.y = matrix(1, 3);
    pose.z = matrix(2, 3);

    // Extract rotation
    double sy = sqrt(matrix(0, 0) * matrix(0, 0) + matrix(1, 0) * matrix(1, 0));

    bool singular = sy < 1e-6; // Near-zero determinant check

    if (!singular) {
        pose.roll = atan2(matrix(2, 1), matrix(2, 2));
        pose.pitch = atan2(-matrix(2, 0), sy);
        pose.yaw = atan2(matrix(1, 0), matrix(0, 0));
    } else {
        pose.roll = atan2(-matrix(1, 2), matrix(1, 1));
        pose.pitch = atan2(-matrix(2, 0), sy);
        pose.yaw = 0;
    }
    return pose;
}

/*----------------------------------------------------------------------------------------------*/

Matrix6f J1_n(pointcloud_clustering::positionRPY ab, pointcloud_clustering::positionRPY bc){ // Jacobian 
  Matrix6f result; result = result.Identity();
  Matrix4f H1 = createHomogMatr(ab);
  Matrix4f H2 = createHomogMatr(bc);
  
  pointcloud_clustering::positionRPY ac;
  
  ac = Comp(ab, bc);
  
  result(0,3) = ab.y - ac.y;     result(0,4) = (ac.z - ab.z)*cos(ab.yaw);                                                                   result(0,5) = H1(0,2)*bc.y - H1(0,1)*bc.z;
  result(1,3) = ac.x - ab.x;     result(1,4) = (ac.z - ab.z)*sin(ab.yaw);                                                                   result(1,5) = H1(1,2)*bc.y - H1(1,1)*bc.z;
  result(2,3) = 0.0;             result(2,4) = -bc.x*cos(ab.pitch) - bc.y*sin(ab.pitch)*sin(ab.roll) - bc.z*sin(ab.pitch)*cos(ab.roll);     result(2,5) = H1(2,2)*bc.y - H1(2,1)*bc.z;
  
  result(3,3) = 1.0;             result(3,4) = sin(ac.pitch)*sin(ac.yaw-ab.yaw)/cos(ac.pitch);            result(3,5) = (H2(0,1)*sin(ac.roll)+H2(0,2)*cos(ac.roll))/cos(ac.pitch);
  result(4,3) = 0.0;             result(4,4) = cos(ac.yaw-ab.yaw);                                        result(4,5) = -cos(ab.pitch)*sin(ac.yaw-ab.yaw);
  result(5,3) = 0.0;             result(5,4) = sin(ac.yaw-ab.yaw)/cos(ac.pitch);                          result(5,5) = cos(ab.pitch)*cos(ac.yaw-ab.yaw)/cos(ac.pitch);
  
  return result;
}

/*----------------------------------------------------------------------------------------------*/

Matrix6f J2_n(pointcloud_clustering::positionRPY ab, pointcloud_clustering::positionRPY bc){
  Matrix6f result = result.Zero();
  Matrix4f H1 = createHomogMatr(ab);
  pointcloud_clustering::positionRPY ac;
  
  ac = Comp(ab, bc);
  
  result(0,0) = cos(ab.yaw)*cos(ab.pitch);  result(0,1) = cos(ab.yaw)*sin(ab.pitch)*sin(ab.roll)-sin(ab.yaw)*cos(ab.pitch);   result(0,2) = cos(ab.yaw)*sin(ab.pitch)*cos(ab.roll)+sin(ab.yaw)*sin(ab.pitch);
  result(1,0) = sin(ab.yaw)*cos(ab.pitch);  result(1,1) = sin(ab.yaw)*sin(ab.pitch)*sin(ab.roll)+cos(ab.yaw)*cos(ab.roll);   result(1,2) = sin(ab.yaw)*sin(ab.pitch)*cos(ab.roll)-cos(ab.yaw)*sin(ab.roll);
  result(2,0) = -sin(ab.pitch);             result(2,1) = cos(ab.pitch)*sin(ab.roll);     result(2,2) = cos(ab.pitch)*cos(ab.roll);
  
  result(3,3) = cos(bc.pitch)*cos(ac.roll-bc.roll)/cos(ac.pitch);                  result(3,4) = sin(ac.roll-bc.roll);                                   result(3,5) = 0.0;
  result(4,3) = -cos(bc.pitch)*sin(ac.roll-bc.roll);                               result(4,4) = cos(ac.roll-bc.roll);                                   result(4,5) = 0.0;
  result(5,3) = (H1(0,2)*cos(ac.yaw) + H1(1,2)*sin(ac.yaw))/cos(ac.pitch);         result(5,4) = sin(ac.pitch)*sin(ac.roll-bc.roll)/cos(ac.pitch);       result(5,5) = 1.0;
  
  return result;
}

/*----------------------------------------------------------------------------------------------*/

float mahalanobisDistance(const MatrixXf& h, const MatrixXf& S){ 
  
//  Vector6f diff;
//  float result;
  MatrixXf hTSih(1, 1); hTSih = hTSih.Zero(1, 1);
//  std::cout << "h:" << std::endl << h << std::endl;
//  std::cout << "S:" << std::endl << S << std::endl;
//  diff = RPY2Vec(map.position) - RPY2Vec(candidate.position);
  hTSih = h.transpose()*S.inverse()*h;
//  hTSih = hTSih.sqrt();
//  std::cout<< "fun mahalan: " << (h.transpose()*S.inverse()*h) << std::endl;
  
  return hTSih(0, 0);
}

// /*----------------------------------------------------------------------------------------------*/

Vector4f computeInnovation(pointcloud_clustering::positionRPY obs, pointcloud_clustering::positionRPY map_landmark, Matrix <float, 4, 6> B){ // Innovation step of EKF
  // Innovation vector indicates de distance between the measured observation referenced to the real one (map landmarks)
  
  return(B * RPY2Vec(Comp(Inv(map_landmark), obs)));
}

// /*----------------------------------------------------------------------------------------------*/

pointcloud_clustering::positionRPY Inv(pointcloud_clustering::positionRPY position){ // Same of createHomogMatrInv?
    Matrix4f H = createHomogMatr(position);
    Matrix4f H_inv = H.inverse(); // Explicitly assign the inverse
    return coordRPY(H_inv);       // Pass the result to coordRPY
}

// /*----------------------------------------------------------------------------------------------*/

// float AngRango (float ang)
// {
//   float PI = 3.141596;
// 	if (ang > PI)
// 	{
// 		ang=ang-2*PI;
// 		AngRango(ang);
// 	}
// 	if (ang < -PI)
// 	{
// 		ang=2*PI+ang;
// 		AngRango(ang);
// 	}

// 	return ang;
// }

/*----------------------------------------------------------------------------------------------*/

pointcloud_clustering::positionRPY vec2RPY(Vector6f position){ // Conversion Vector6f -> pointcloud_clustering::positionRPY
  pointcloud_clustering::positionRPY result;
  
  result.x = position(0);
  result.y = position(1);
  result.z = position(2);
  result.roll = position(3);
  result.pitch = position(4);
  result.yaw = position(5);
  
  return(result);
}

/*----------------------------------------------------------------------------------------------*/

