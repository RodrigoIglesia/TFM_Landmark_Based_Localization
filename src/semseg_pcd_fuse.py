"""
Main script for semantic segmentation + 3D lidar data data fusion
The objective of this module is to fuse the 3D lidar point cloud segmentation with the semantic segmentation information.
1. Receive point cloud in a rosbag
2. Received segmented label
3. Fuse pointcloud with segmented label
"""

import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pointcloud_callback(msg):
    # Extract point cloud data from ROS message
    pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

    # Convert point cloud data to NumPy array
    points = np.array(list(pc_data))

    # Plot the point cloud in a 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    rospy.init_node('waymo_pointcloud_subscriber', anonymous=True)
    rospy.Subscriber('/waymo_pointcloud', pc2.PointCloud2, pointcloud_callback)

    rospy.spin()

if __name__ == '__main__':
    main()