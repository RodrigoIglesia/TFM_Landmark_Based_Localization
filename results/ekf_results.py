import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)


# Load the files
scene = "individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels"

landmarks_file = os.path.join(src_dir, "results/" + scene + "/202503011722/landmarks_" + scene +".csv")
poses_file = os.path.join(src_dir, "results/" + scene + "/202503011722/poses_" + scene + ".csv")
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
    map_match = frame_data["match_index"].str.replace("[\[\]]", "", regex=True).astype(int)
    observed_coords = frame_data[['Landmark_X', 'Landmark_Y', 'Landmark_Z', 'Landmark_Roll', 'Landmark_Pitch', 'Landmark_Yaw']].values

    observed_x, observed_y = frame_data['Landmark_X'], frame_data['Landmark_Y']
    ax.scatter(observed_x, observed_y, c='green', alpha=0.6)

    # Iterate through each match index
    for i, match in enumerate(map_match):
        if match > -1:
            mam_match_coords = map_coords[match]
            ax.plot(mam_match_coords[0], mam_match_coords[1], 'ro', markersize=8)
            ax.plot([observed_coords[i, 0], mam_match_coords[0]], [observed_coords[i, 1], mam_match_coords[1]], 'k--', alpha=0.5)


noise_text = "position_noise_std=0.1, orientation_noise_std=0.0"
ax.set_title(f'Mahalanobis Distance - {noise_text}')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.legend()
ax.grid(True)
plt.show()
