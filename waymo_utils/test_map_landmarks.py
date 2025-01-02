import open3d as o3d
import numpy as np
import pandas as pd

# Load your data
file_path = "/home/rodrigo/catkin_ws/src/TFM_Landmark_Based_Localization/dataset/pointclouds/pointcloud_concatenatedindividual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.csv"
data = pd.read_csv(file_path, header=None)
data.columns = ['X', 'Y', 'Z']

# Prepare the data for Open3D
points = np.array(data[['X', 'Y', 'Z']])

# Create a point cloud object
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud], window_name="3D Point Cloud", width=800, height=600)
