#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloudSubscriber:
    def __init__(self):
        rospy.init_node('point_cloud_subscriber', anonymous=True)
        rospy.Subscriber("clustered_PointCloud", PointCloud2, self.callback)
        self.points = []

    def callback(self, data):
        rospy.loginfo("Message received in test topic")
        self.points = []
        for point in pc2.read_points(data, skip_nans=True):
            self.points.append([point[0], point[1], point[2]])

    def plot_point_cloud(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        points_array = np.array(self.points)
        ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

if __name__ == '__main__':
    pcl_subscriber = PointCloudSubscriber()
    rospy.spin()
    pcl_subscriber.plot_point_cloud()
