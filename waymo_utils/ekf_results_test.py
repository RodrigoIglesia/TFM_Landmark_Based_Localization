import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)


def create_homogeneous_matrix(position):
    """
    Crea una matriz homogénea 4x4 a partir de una posición (x, y, z, roll, pitch, yaw).
    """
    x, y, z, roll, pitch, yaw = position

    # Rotación en los ejes X, Y y Z
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Matriz de rotación combinada
    R = Rz @ Ry @ Rx

    # Matriz homogénea
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = [x, y, z]
    return H

def inverse_homogeneous_matrix(H):
    """
    Calcula la inversa de una matriz homogénea 4x4.
    """
    R = H[:3, :3]
    t = H[:3, 3]
    H_inv = np.eye(4)
    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ t
    return H_inv

def extract_pose_from_matrix(H):
    """
    Extrae posición (x, y, z) y orientación (roll, pitch, yaw) desde una matriz homogénea.
    """
    x, y, z = H[:3, 3]
    sy = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(H[2, 1], H[2, 2])
        pitch = np.arctan2(-H[2, 0], sy)
        yaw = np.arctan2(H[1, 0], H[0, 0])
    else:
        roll = np.arctan2(-H[1, 2], H[1, 1])
        pitch = np.arctan2(-H[2, 0], sy)
        yaw = 0
    return np.array([x, y, z, roll, pitch, yaw])

def compute_innovation(observation, map_landmark, B):
    """
    Calcula el vector de innovación.
    :param observation: np.array, [x, y, z, roll, pitch, yaw] de la observación.
    :param map_landmark: np.array, [x, y, z, roll, pitch, yaw] del landmark en el mapa.
    :param B: np.array, matriz de proyección 4x6 para reducir dimensiones.
    :return: np.array, vector de innovación 4x1.
    """
    # Matrices homogéneas
    H_map_landmark = create_homogeneous_matrix(map_landmark)
    inv_map_landmark = extract_pose_from_matrix(inverse_homogeneous_matrix(H_map_landmark))
    H_map_landmark_inv  = create_homogeneous_matrix(map_landmark)
    H_obs =  create_homogeneous_matrix(observation)

    # Transformar la observación al marco del landmark
    H_relative = H_map_landmark_inv @ H_obs

    relative_pose = extract_pose_from_matrix(H_relative)

    # Aplicar la matriz de proyección B para obtener la innovación
    innovation = B * relative_pose
    return innovation


import numpy as np

def J1_n(ab, bc):
    """
    Calcula la Jacobiana J1_n (respecto al estado acumulado).
    :param ab: np.array, pose acumulada (x, y, z, roll, pitch, yaw).
    :param bc: np.array, incremento (x, y, z, roll, pitch, yaw).
    :return: np.array, matriz 6x6 Jacobiana.
    """
    H1 = create_homogeneous_matrix(ab)
    H2 = create_homogeneous_matrix(bc)
    ac = extract_pose_from_matrix(H1 @ H2)  # Pose acumulada

    J1 = np.eye(6)

    # Componentes de la matriz de rotación H1
    J1[0, 3] = ab[1] - ac[1]
    J1[0, 4] = (ac[2] - ab[2]) * np.cos(ab[5])
    J1[0, 5] = H1[0, 2] * bc[1] - H1[0, 1] * bc[2]

    J1[1, 3] = ac[0] - ab[0]
    J1[1, 4] = (ac[2] - ab[2]) * np.sin(ab[5])
    J1[1, 5] = H1[1, 2] * bc[1] - H1[1, 1] * bc[2]

    J1[2, 4] = -bc[0] * np.cos(ab[4]) - bc[1] * np.sin(ab[4]) * np.sin(ab[3]) - bc[2] * np.sin(ab[4]) * np.cos(ab[3])
    J1[2, 5] = H1[2, 2] * bc[1] - H1[2, 1] * bc[2]

    J1[3, 4] = np.sin(ac[4]) * np.sin(ac[5] - ab[5]) / np.cos(ac[4])
    J1[3, 5] = (H2[0, 1] * np.sin(ac[3]) + H2[0, 2] * np.cos(ac[3])) / np.cos(ac[4])

    J1[4, 4] = np.cos(ac[5] - ab[5])
    J1[4, 5] = -np.cos(ab[4]) * np.sin(ac[5] - ab[5])

    J1[5, 4] = np.sin(ac[5] - ab[5]) / np.cos(ac[4])
    J1[5, 5] = np.cos(ab[4]) * np.cos(ac[5] - ab[5]) / np.cos(ac[4])

    return J1

def J2_n(ab, bc):
    """
    Calcula la Jacobiana J2_n (respecto al incremento).
    :param ab: np.array, pose acumulada (x, y, z, roll, pitch, yaw).
    :param bc: np.array, incremento (x, y, z, roll, pitch, yaw).
    :return: np.array, matriz 6x6 Jacobiana.
    """
    H1 = create_homogeneous_matrix(ab)
    ac = extract_pose_from_matrix(H1 @ create_homogeneous_matrix(bc))  # Pose acumulada

    J2 = np.zeros((6, 6))

    J2[:3, :3] = np.array([
        [np.cos(ab[5]) * np.cos(ab[4]), np.cos(ab[5]) * np.sin(ab[4]) * np.sin(ab[3]) - np.sin(ab[5]) * np.cos(ab[3]), np.cos(ab[5]) * np.sin(ab[4]) * np.cos(ab[3]) + np.sin(ab[5]) * np.sin(ab[3])],
        [np.sin(ab[5]) * np.cos(ab[4]), np.sin(ab[5]) * np.sin(ab[4]) * np.sin(ab[3]) + np.cos(ab[5]) * np.cos(ab[3]), np.sin(ab[5]) * np.sin(ab[4]) * np.cos(ab[3]) - np.cos(ab[5]) * np.sin(ab[3])],
        [-np.sin(ab[4]), np.cos(ab[4]) * np.sin(ab[3]), np.cos(ab[4]) * np.cos(ab[3])]
    ])

    J2[3:, 3:] = np.eye(3)

    return J2

def mahalanobis_distance(h, S):
    """
    Computes the Mahalanobis distance.
    :param h: np.array, innovation vector (1D array of shape (n,))
    :param S: np.array, covariance matrix (2D array of shape (n, n))
    :return: float, Mahalanobis distance
    """
    # Ensure inputs are NumPy arrays
    h = np.asarray(h)
    S = np.asarray(S)

    # Check dimensions
    if h.ndim != 1 or S.ndim != 2:
        raise ValueError("h must be a 1D array and S must be a 2D array.")
    if h.shape[0] != S.shape[0] or S.shape[0] != S.shape[1]:
        raise ValueError("Dimensions of h and S must match.")

    # Compute Mahalanobis distance
    S_inv = np.linalg.pinv(S) if np.linalg.det(S) < 1e-6 else np.linalg.inv(S)  # Handle non-invertible cases
    hTSih = h.T @ S_inv @ h
    return np.sqrt(hTSih)



# Cargar los archivos
landmarks_file = os.path.join(src_dir, "results/landmarks_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")
poses_file = os.path.join(src_dir, "results/poses_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")
signs_map_file = os.path.join(src_dir, "pointcloud_clustering/map/signs_map_features_individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.csv")

landmarks_data = pd.read_csv(landmarks_file)
poses_data = pd.read_csv(poses_file)
signs_map_data = pd.read_csv(signs_map_file, header=None)

# Covarianza asociada con las observaciones y el mapa
P = np.eye(6)  # Matriz de covarianza asociada con las poses
R = np.eye(6)  # Ruido del sensor (covarianza de las observaciones)
B = np.array([
    [1, 0, 0, 0, 0, 0],  # x
    [0, 1, 0, 0, 0, 0],  # y
    [0, 0, 0, 1, 0, 0],  # roll
    [0, 0, 0, 0, 1, 0]   # pitch
], dtype=float)

# Coordenadas del mapa (landmarks del mapa)
map_coords = signs_map_data[[0, 1, 2]].values  # Pose en el mapa

# Obtener los frames únicos
unique_frames = landmarks_data['frame'].unique()

# Calcular la matriz de covarianza y su inversa para el cálculo de Mahalanobis
cov_matrix = np.cov(map_coords, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Iterar sobre cada frame para crear un gráfico individual
for frame in unique_frames:
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filtrar landmarks observados para el frame actual
    frame_data = landmarks_data[landmarks_data['frame'] == frame]
    observed_coords = frame_data[['Landmark_X', 'Landmark_Y', 'Landmark_Z', 'Landmark_Roll', 'Landmark_Pitch', 'Landmark_Yaw']].values
    current_pose_data = poses_data[poses_data['frame'] == frame]
    kalmanPose = np.array([
        current_pose_data['real_x'].iloc[0],
        current_pose_data['real_y'].iloc[0],
        current_pose_data['real_z'].iloc[0],
        current_pose_data['real_roll'].iloc[0],
        current_pose_data['real_pitch'].iloc[0],
        current_pose_data['real_yaw'].iloc[0]
    ])

    # 1. Ploteo de landmarks del mapa con círculos
    map_x = map_coords[:, 0]
    map_y = map_coords[:, 1]
    for x, y in zip(map_x, map_y):
        ax.plot(x, y, 'o', markersize=8, markeredgecolor='blue', markerfacecolor='none', label='Landmarks (Mapa)' if 'Landmarks (Mapa)' not in ax.get_legend_handles_labels()[1] else "")

    # 2. Ploteo de landmarks observados
    observed_x = frame_data['Landmark_X']
    observed_y = frame_data['Landmark_Y']
    scatter = ax.scatter(observed_x, observed_y, c='green', label='Landmarks Observados', alpha=0.6)

    # Calcular y mostrar la distancia de Mahalanobis
    for obs in observed_coords:
        distances = []
        for landmark in map_coords:
            landmark = np.concatenate([landmark, [0, 0, 0]])

            # Innovación: diferencia entre observación y landmark
            h_ij = compute_innovation(obs, landmark, B)
            # Calcular Jacobianas
            
            H_z_ij = B @ J2_n(extract_pose_from_matrix(inverse_homogeneous_matrix(create_homogeneous_matrix(landmark))), obs) @ J2_n(kalmanPose, obs)
            H_x_ij = B @ J2_n(extract_pose_from_matrix(inverse_homogeneous_matrix(create_homogeneous_matrix(landmark))), obs) @ J1_n(kalmanPose, obs)

            # Innovación de la covarianza
            S_ij = H_x_ij @ P @ H_x_ij.T + H_z_ij @ R @ H_z_ij.T
            # Distancia de Mahalanobis
            distance_mahalanobis =  mahalanobis_distance(h_ij, S_ij)
            print(distance_mahalanobis)
            distances.append(distance_mahalanobis)

        # distances = [distance.mahalanobis(obs, point, inv_cov_matrix) for point in map_coords]
        closest_idx = np.argmin(distances)
        closest_point = map_coords[closest_idx]
        min_distance = distances[closest_idx]
        min_distance = float(min_distance)

        # Dibujar una línea entre el punto observado y el punto del mapa más cercano
        ax.plot([obs[0], closest_point[0]], [obs[1], closest_point[1]], 'k--', alpha=0.5)
        
        # Mostrar la distancia de Mahalanobis en el gráfico
        ax.text((obs[0] + closest_point[0]) / 2, (obs[1] + closest_point[1]) / 2,
                f"{min_distance:.2f}", fontsize=8, color='red', alpha=0.7)

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

    # Ajustes del gráfico
    ax.set_title(f'Mapa, Observaciones y Poses (Frame {frame}) - Distancias de Mahalanobis')
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.legend()
    ax.grid(True)

    # Mostrar el gráfico
    plt.show()
