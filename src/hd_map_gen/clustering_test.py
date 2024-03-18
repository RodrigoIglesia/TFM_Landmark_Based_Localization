import os
import sys
import numpy as np

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *

if __name__ == "__main__":
    point_cloud = np.loadtxt('dataset/pointcloud_concatenated.csv', delimiter=',')

    clustered_pointcloud, cluster_labels = cluster_pointcloud(point_cloud)

    show_point_cloud_with_labels(point_cloud, cluster_labels)
    for cluster in clustered_pointcloud:
        # Get the centroid of each cluster of the pointcloud
        cluster_centroid = get_cluster_centroid(cluster)



