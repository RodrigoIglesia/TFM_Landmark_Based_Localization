import os
import sys
import pathlib
import json
import plotly.graph_objs as go

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import plot_maps
from map_pointcloud_concatenated import *

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *

def save_map_json(map_dict, filepath):
    # Use json.dump to save the dictionary to a JSON file
    with open(filepath, 'w') as json_file:
        print('Saving: ', filepath)
        json.dump(map_dict, json_file)


if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    print('-------------------------------------\n')
    print('-------------------------------------\n')

    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        frame_index = 0
        for frame in load_frame(scene_path):
            if hasattr(frame, 'map_features') and frame.map_features:
                # Retrieve map_feature in the firts frame
                map_features = frame.map_features

            break
        point_cloud = np.loadtxt('dataset/pointcloud_concatenated.csv', delimiter=',')

        clustered_pointcloud, cluster_labels = cluster_pointcloud(point_cloud)

        # show_point_cloud_with_labels(point_cloud, cluster_labels)

        sign_id = map_features[-1].id
        for cluster in clustered_pointcloud:
            # Get the centroid of each cluster of the pointcloud
            cluster_centroid = get_cluster_centroid(cluster)
            signs_map_feature = add_sign_to_map(map_features, cluster_centroid, sign_id)
            sign_id += 1

        # figure = plot_maps.plot_map_features(signs_map_feature)
        # figure.show()
        plot_pointcloud_on_map(signs_map_feature, point_cloud, cluster_labels)

