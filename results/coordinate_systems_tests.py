import open3d as o3d
import pandas as pd
import numpy as np
import os
import sys

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

# Función para crear ejes de coordenadas con orientación
def create_coordinate_axes_with_orientation(position, orientation, size=0.1):
    """Crea un frame de coordenadas con orientación especificada usando ángulos de Euler."""
    position = np.array(position, dtype=np.float64)  # Asegurar que la posición sea float64
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=position)
    
    # Convertir los ángulos de Euler a matriz de rotación
    rx, ry, rz = map(float, orientation)  # Asegurar que sean floats
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])  # Asumimos ángulos en radianes
    axes.rotate(rotation_matrix, center=position)
    return axes

# Cargar datos de los CSV
scene = "individual_files_validation_segment-10289507859301986274_4200_000_4220_000_with_camera_labels"
date = "202503151657"

landmarks_file = os.path.join(src_dir, "results/" + scene + "/" + date, "landmarks_" + scene + ".csv")
map_pointcloud_file = os.path.join(src_dir, "dataset/pointclouds/pointcloud_concatenated" + scene + ".csv")
signs_map_file = os.path.join(src_dir, "pointcloud_clustering/map/signs_map_features_" + scene + ".csv")

# Leer los CSV
map_pointcloud = pd.read_csv(map_pointcloud_file)
landmarks = pd.read_csv(landmarks_file)
signs_map = pd.read_csv(signs_map_file, header=None)

# Limpiar los valores en 'frame' y 'match_index' eliminando corchetes y convirtiéndolos a enteros
landmarks["frame"] = landmarks["frame"].astype(str).str.extract(r'(-?\d+)').astype(int)
landmarks["match_index"] = landmarks["match_index"].astype(str).str.extract(r'(-?\d+)').astype(int)

landmarks_match = landmarks[landmarks["match_index"] != -1]
print("Landmarks detectados con match:")
print(landmarks_match)

# Crear la nube de puntos de map_pointcloud_features
points = map_pointcloud.iloc[:, :3].values  # Se asume que las primeras tres columnas son X, Y, Z
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Coordenadas de los landmarks en el mapa
map_coords = signs_map[[0, 1, 2]].values

# Crear ejes de coordenadas para cada landmark en signs_map
axes_list = []
labels_list = []

for index, row in signs_map.iterrows():
    position = row.iloc[0:3].astype(float).values  # Se asume que X, Y, Z están en las columnas 0, 1 y 2
    euler_angles = np.array([0, 0, 0], dtype=np.float64)  # Si no hay orientación en signs_map, se usa 0
    axes = create_coordinate_axes_with_orientation(position, euler_angles, size=2.0)
    axes_list.append(axes)

# for index, row in landmarks_match.iterrows():
#     position = row.iloc[0:3].astype(float).values  # Se asume que X, Y, Z están en las columnas 0, 1 y 2
#     euler_angles = np.array([0, 0, 0], dtype=np.float64)  # Si no hay orientación en signs_map, se usa 0
#     axes = create_coordinate_axes_with_orientation(position, euler_angles, size=2.0)
#     axes_list.append(axes)

# Añadir eje de coordenadas en el origen
origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0, 0, 0], dtype=np.float64))

# Mostrar la escena con los ejes de coordenadas y etiquetas
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="MapLandmarks", width=800, height=600)
vis.add_geometry(point_cloud)
# vis.add_geometry(origin_axes)
for axes in axes_list:
    vis.add_geometry(axes)



vis.run()
vis.destroy_window()
