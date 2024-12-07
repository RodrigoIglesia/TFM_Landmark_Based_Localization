import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

# Cargar los archivos
landmarks_file = os.path.join(src_dir, "results/landmarks_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")
poses_file = os.path.join(src_dir,"results/poses_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")
signs_map_file = os.path.join(src_dir,"pointcloud_clustering/map/signs_map_features_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")

landmarks_data = pd.read_csv(landmarks_file)
poses_data = pd.read_csv(poses_file)
signs_map_data = pd.read_csv(signs_map_file, header=None)

# Configuración para el plot
fig, ax = plt.subplots(figsize=(12, 8))

# 1. Ploteo de landmarks del mapa con círculos
map_x = signs_map_data[0]
map_y = signs_map_data[1]
for x, y in zip(map_x, map_y):
    ax.plot(x, y, 'o', markersize=8, markeredgecolor='blue', markerfacecolor='none', label='Landmarks (Mapa)' if 'Landmarks (Mapa)' not in ax.get_legend_handles_labels()[1] else "")

# 2. Ploteo de landmarks observados
observed_x = landmarks_data['Landmark_X']
observed_y = landmarks_data['Landmark_Y']
frames = landmarks_data['frame']
scatter = ax.scatter(observed_x, observed_y, c='green', label='Landmarks Observados', alpha=0.6)

# Agregar etiquetas de frame para cada landmark observado
for i, txt in enumerate(frames):
    ax.annotate(txt, (observed_x.iloc[i], observed_y.iloc[i]), fontsize=8, alpha=0.8)

# 3. Ploteo de poses
real_x = poses_data['real_x']
real_y = poses_data['real_y']
odom_x = poses_data['odometry_x']
odom_y = poses_data['odometry_y']
corrected_x = poses_data['corrected_x']
corrected_y = poses_data['corrected_y']

# Pose real
ax.plot(real_x, real_y, '-r', label='Pose Real')
# Pose odométrica (si está disponible)
if not odom_x.isnull().all():
    ax.plot(odom_x, odom_y, '--g', label='Pose Odómétrica')
# Pose corregida (si está disponible)
if not corrected_x.isnull().all():
    ax.plot(corrected_x, corrected_y, ':b', label='Pose Corregida')

# Ajustes finales del gráfico
ax.set_title('Mapa, Observaciones y Poses (Landmarks con Círculos)')
ax.set_xlabel('Coordenada X')
ax.set_ylabel('Coordenada Y')
ax.legend()
ax.grid(True)

# Mostrar el gráfico
plt.show()
