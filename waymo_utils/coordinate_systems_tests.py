import open3d as o3d
import pandas as pd
import numpy as np
import os
import sys

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

# Cargar datos de los CSV
landmarks_file = os.path.join(src_dir, "results/landmarks_individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.csv")
poses_file = os.path.join(src_dir, "results/poses_individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.csv")
signs_map_file = os.path.join(src_dir, "dataset/pointclouds/pointcloud_concatenatedindividual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.csv")

# Leer los CSV
signs_map = pd.read_csv(signs_map_file)
landmarks = pd.read_csv(landmarks_file)
print(landmarks.head())

# Crear la nube de puntos de signs_map_features
points = signs_map.iloc[:, :3].values  # Se asume que las primeras tres columnas son X, Y, Z
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Función para crear ejes de coordenadas con orientación
def create_coordinate_axes_with_orientation(position, orientation, size=0.1):
    """Crea un frame de coordenadas con orientación especificada usando ángulos de Euler."""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=position)
    # Convertir los ángulos de Euler a matriz de rotación
    rx, ry, rz = orientation
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])  # Asumimos ángulos en radianes
    axes.rotate(rotation_matrix, center=position)
    return axes

# Crear ejes para landmarks_individual con orientación
frame_to_visualize = '[3]'  # Cambia este valor para seleccionar el frame
landmarks_filtered = landmarks[landmarks.iloc[:, 0] == frame_to_visualize]
axes_list = []

for _, row in landmarks_filtered.iterrows():
    position = row.iloc[1:4].values  # Se asume que las columnas 2, 3 y 4 son X, Y, Z
    euler_angles = row.iloc[4:7].values  # Se asume que las columnas 5, 6 y 7 son rx, ry, rz en radianes
    axes = create_coordinate_axes_with_orientation(position, euler_angles, size=2.0)
    axes_list.append(axes)

# Añadir eje de coordenadas en el origen
origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

# Crear la visualización
o3d.visualization.draw_geometries([point_cloud, origin_axes, *axes_list],
                                  window_name=f"Frame {frame_to_visualize}",
                                  width=800,
                                  height=600)
