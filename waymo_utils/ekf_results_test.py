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
    x, y, z, roll, pitch, yaw = position
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = [x, y, z]
    return H

def inverse_homogeneous_matrix(H):
    R = H[:3, :3]
    t = H[:3, 3]
    H_inv = np.eye(4)
    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ t
    return H_inv

def mahalanobis_distance(h, S):
    h = np.asarray(h)
    S = np.asarray(S)
    if h.ndim != 1 or S.ndim != 2:
        raise ValueError("h must be 1D and S must be 2D.")
    if h.shape[0] != S.shape[0] or S.shape[0] != S.shape[1]:
        raise ValueError("Dimensions of h and S must match.")
    S_inv = np.linalg.pinv(S) if np.linalg.det(S) < 1e-6 else np.linalg.inv(S)
    hTSih = h.T @ S_inv @ h
    return np.sqrt(hTSih)

def compute_mahalanobis_distances(observation, map_points, inv_cov_matrix):
    distances = []
    for point in map_points:
        diff = observation[:3] - point
        distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)
        distances.append(distance)
    return distances

# Load the files
scene = "individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels"

landmarks_file = os.path.join(src_dir, "results/" + scene + "/190120251935/landmarks_" + scene +".csv")
poses_file = os.path.join(src_dir, "results/" + scene + "/190120251935/poses_" + scene + ".csv")
signs_map_file = os.path.join(src_dir, "pointcloud_clustering/map/signs_map_features_" + scene +".csv")

landmarks_data = pd.read_csv(landmarks_file)
poses_data = pd.read_csv(poses_file)
signs_map_data = pd.read_csv(signs_map_file, header=None)

map_coords = signs_map_data[[0, 1, 2]].values
cov_matrix = np.cov(map_coords, rowvar=False)
inv_cov_matrix = np.linalg.pinv(cov_matrix)

unique_frames = landmarks_data['frame'].unique()
fig, ax = plt.subplots(figsize=(12, 8))

# Graficar el mapa y definir etiquetas fuera del bucle
map_x, map_y = map_coords[:, 0], map_coords[:, 1]
ax.plot(map_x, map_y, 'o', markersize=8, markeredgecolor='blue', markerfacecolor='none', label='Landmarks (Map)')

real_x, real_z = poses_data['real_x'], poses_data['real_y']
ax.plot(real_x, real_z, '-r', label='Real Pose')
odom_x, odom_z = poses_data['odometry_x'], poses_data['odometry_y']
ax.plot(odom_x, odom_z, '-g', label='Odometry Pose')
corrected_x, corrected_z = poses_data['corrected_x'], poses_data['corrected_y']
ax.plot(corrected_x, corrected_z, '-b', label='Corrected Pose')

for frame in unique_frames:
    frame_data = landmarks_data[landmarks_data['frame'] == frame]
    observed_coords = frame_data[['Landmark_X', 'Landmark_Y', 'Landmark_Z', 'Landmark_Roll', 'Landmark_Pitch', 'Landmark_Yaw']].values

    for obs in observed_coords:
        distances = compute_mahalanobis_distances(obs, map_coords, inv_cov_matrix)
        closest_idx = np.argmin(distances)
        closest_point = map_coords[closest_idx]
        min_distance = distances[closest_idx]

        ax.plot([obs[0], closest_point[0]], [obs[1], closest_point[1]], 'k--', alpha=0.5)
        # ax.text((obs[0] + closest_point[0]) / 2, (obs[1] + closest_point[1]) / 2,
        #         f"{min_distance:.2f}", fontsize=8, color='red', alpha=0.7)

    observed_x, observed_y = frame_data['Landmark_X'], frame_data['Landmark_Y']
    ax.scatter(observed_x, observed_y, c='green', alpha=0.6)

noise_text = "position_noise_std=0.1, orientation_noise_std=0.0"
ax.set_title(f'Mahalanobis Distance - {noise_text}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True)
plt.show()
