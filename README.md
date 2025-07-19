# WayLoc: Modular Vehicle Localization System Using ROS and Waymo Dataset

**WayLoc** is a modular localization system for autonomous vehicles, developed using **ROS**, **Python**, and **C++**. It integrates multiple perception (LiDAR and High-Resolution cameras) and data fusion modules leveraging the **Waymo Open Dataset** to estimate the vehicle's position in complex urban environments.

---

## 🚗 Project Overview

This project aims to implement a robust localization pipeline for autonomous vehicles using a combination of:

- **Landmark detection from LiDAR point clouds**
- **Semantic segmentation from RGB images**
- **Data association techniques**
- **Sensor fusion with an Extended Kalman Filter (EKF)**
- **Testing and benchmarking utilities**

Each component is developed as an independent ROS package and follows a modular, extensible design.

---

## 📦 Project Structure

```
WaymoLoc/
├── hd_map_gen/ # HD map generation script
├── pointcloud_clustering/ # Main module for landmark detection and localization
│ ├── config/ # Configuration files
│ ├── include/ # C++ header files
│ ├── launch/ # ROS launch files
│ ├── msg/ # Custom ROS messages
│ ├── srv/ # Custom ROS services
│ ├── src/ # Core implementations (Python/C++)
├── waymo_utils/ # Tools for parsing Waymo dataset
├── dataset/ # Input scenes from Waymo Open Dataset
├── models/ # Pre-trained models and checkpoints
├── logs/ # System execution logs
├── results/ # Output data and evaluation results
├── uml/ # System modelling UML files. Developed in PlantUML.
├── waymoloc_full.rviz # Rviz config for full localization system testing
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- ROS (Melodic/Noetic recommended)
- Python 3.6+
- C++14
- [Waymo Open Dataset](https://waymo.com/open/)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/wayloc.git
    cd wayloc
    ```
2. Install dependencies:
    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    pip install -r requirements.txt
    ```
3. Build the workspace:
    ```bash
    catkin_make
    source devel/setup.bash
    ```

---

## 🛠️ Usage

- Launch the full localization pipeline:
  ```bash
  roslaunch launch/waymoloc_full.launch
  ```
- Run individual modules for testing and development.
<p align="center">
  <img src="https://github.com/user-attachments/assets/55a763a3-1a5c-4e70-9922-40c5e77a0854" alt="Tests_correction_scene" width="600"/>
  <img src="https://github.com/user-attachments/assets/7e2d1c5f-e8ae-4b33-a419-743dad1489a2" alt="EKF_mal_ajustado" width="600"/>
</p>



---

## 📖 Documentation

See the [docs/](docs/) directory for detailed module descriptions, usage examples, and API references.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.
