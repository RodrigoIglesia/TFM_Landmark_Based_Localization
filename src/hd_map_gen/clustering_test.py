import os
import sys
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *

def get_differentiated_colors(numbers):
    # Choose a colormap (you can change this to any other colormap available in matplotlib)
    colormap = cm.get_cmap('hsv')

    # Normalize the numbers to map them to the colormap range
    normalize = Normalize(vmin=min(numbers), vmax=max(numbers))

    # Map each number to a color in the chosen colormap
    colors = [colormap(normalize(num)) for num in numbers]

    return colors

if __name__ == "__main__":
    # load pointclous saved in csv file
    point_cloud_path = 'dataset/pointclouds/'

    for csv_file in os.listdir(point_cloud_path):
        print(csv_file)
        point_cloud = np.loadtxt(os.path.join(point_cloud_path, csv_file), delimiter=',')
        origin = np.mean(point_cloud, axis=0)
        print(origin)
        # colors = get_differentiated_colors(cluster_labels)

        # Plot Map and PointCloud aligned with the map data.
        # plt.figure()

        # plt.subplot(1, 2, 1)
        # plt.scatter(point_cloud[:,0], point_cloud[:,1], color='blue')

        clustered_pointcloud, cluster_labels = cluster_pointcloud(point_cloud)

        colors = get_differentiated_colors(cluster_labels)

        # Plot Map and PointCloud aligned with the map data.
        # plt.subplot(1, 2, 2)
        # plt.scatter(point_cloud[:,0], point_cloud[:,1], color=colors)
        # plt.show()

        show_point_cloud_with_labels(point_cloud, cluster_labels)
        
        for cluster in clustered_pointcloud:
            # Get the centroid of each cluster of the pointcloud
            cluster_centroid = get_cluster_centroid(cluster)




