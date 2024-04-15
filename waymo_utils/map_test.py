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

from WaymoParser import *
from waymo_3d_parser import *

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.hd_map_gen.map_pointcloud_concatenated import *



if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    print('-------------------------------------\n')
    print('-------------------------------------\n')

    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        frame_index = 0
        for frame in load_frame(scene_path):
            if hasattr(frame, 'map_features') and frame.map_features:
                # Retrieve map_feature in the firts frame
                map_features = frame.map_features
            break
        save_protobuf_features(map_features, "dataset/hd_maps/map_features_" + os.path.splitext(os.path.basename(scene_path))[0] + '.json')
        point_cloud = np.loadtxt('dataset/pointcloud_concatenated.csv', delimiter=',')

        clustered_pointcloud, cluster_labels = cluster_pointcloud(point_cloud)

        # show_point_cloud_with_labels(point_cloud, cluster_labels)

        sign_id = map_features[-1].id
        for cluster in clustered_pointcloud:
            # Get the centroid of each cluster of the pointcloud
            cluster_centroid = get_cluster_centroid(cluster)
            add_sign_to_map(map_features, cluster_centroid, sign_id)

            sign_id += 1

        save_protobuf_features(map_features, "dataset/hd_maps/signs_map_features_" + os.path.splitext(os.path.basename(scene_path))[0] + '.json')
        for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
            output_filename = 'dataset/waymo_map_scene_mod/output' + os.path.basename(scene_path)
            writer = tf.io.TFRecordWriter(output_filename)
            for frame in load_frame(scene_path):
                if hasattr(frame, 'map_features') and frame.map_features:
                    # Retrieve map_feature in the firts 
                    del frame.map_features[:]

                    # Append the new map_features object to the cleared list
                    for feature in map_features:
                        frame.map_features.append(feature)
                    print(frame.map_features)
                serialized_frame = frame.SerializeToString()
                writer.write(serialized_frame)

            # Close the writer
            writer.close()


        # figure = plot_maps.plot_map_features(signs_map_feature)
        # figure.show()
        # plot_pointcloud_on_map(map_features, point_cloud, cluster_labels)

