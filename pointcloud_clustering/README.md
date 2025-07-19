# PointCloud Clustering Module

This ROS package provides core functionality for landmark detection and clustering from LiDAR point clouds, supporting vehicle localization pipelines using the Waymo Open Dataset.

## 🚀 Features

- Cropping, downsampling, and ground extraction of point clouds
- Euclidean clustering for landmark (e.g., traffic sign) detection
- ROS service interface for clustering and landmark detection
- Debug publishers for visualization in RViz
- Configurable via `.conf` files in the `config/` directory

## 📂 Structure

- `src/` – Core C++ and Python implementations ([clustering.cpp](src/clustering.cpp), [landmark_detection.py](src/landmark_detection.py))
- `config/` – Configuration files for clustering and data fusion
- `msg/` & `srv/` – Custom ROS messages and services
- `launch/` – Example launch files
- `map/` – Example output CSVs with detected landmarks

## ⚙️ Usage

1. Build the package:
    ```bash
    catkin_make
    source devel/setup.bash
    ```
2. Launch clustering node:
    ```bash
    roslaunch pointcloud_clustering pc_clustering_v1.launch
    ```
3. Visualize results in RViz using the provided topics.

## 📝 Configuration

Edit parameters in `config/clustering_config.conf` and `config/data_fusion_config.conf` to adjust cropping, clustering, and fusion settings.

## 📄 Dependencies

- ROS (Melodic/Noetic)
- PCL
- Boost (program_options)
- sensor_msgs, geometry_msgs, visualization_msgs

See the main [WayLoc README](../../README.md) for full