cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_clustering)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(PCL 1.2 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  rospy
  std_msgs
  rosbag
  geometry_msgs
  nav_msgs
  visualization_msgs
  message_generation
  message_runtime
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  CameraProj.msg
  positionRPY.msg
  observationRPY.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  clustering_srv.srv
  landmark_detection_srv.srv
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

## The catkin_package macro generates cmake config files for your package
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime sensor_msgs geometry_msgs nav_msgs visualization_msgs
  INCLUDE_DIRS include
)

include_directories(
 include
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(clustering src/clustering.cpp)
add_executable(data_fusion src/data_fusion.cpp)
target_link_libraries(clustering ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${rosbag_LIBRARIES})
target_link_libraries(data_fusion ${catkin_LIBRARIES})

add_dependencies(clustering ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
