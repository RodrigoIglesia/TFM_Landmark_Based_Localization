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


def add_sign_to_map(map_features, sign_coords):
    # Create a new map features object to insert the sign there
    new_map_feature = map_pb2.MapFeature()

    # Create a new Driveway message and populate its polygons
    sign_message = new_map_feature.stop_sign
    sign_message.position.x = sign_coords[0]
    sign_message.position.y = sign_coords[1]
    sign_message.position.z = sign_coords[2]

    # Append the new map feature object in the existing map feature
    map_features.append(new_map_feature)


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

                add_sign_to_map(map_features, )
                print(map_features)

                # Plot the point cloud for this frame aligned with the map data.
                figure = plot_maps.plot_map_features(map_features)
                figure.show()

                # Save new frame to tfrecord serialized file
                # output_tfrecord_path = f'hd_map_mod_{scene_index}_{frame_index}.tfrecord'
                # with tf.io.TFRecordWriter(output_path) as writer:
                #     writer.write(frame.SerializeToString())
                # print('Saved TFRecord:', output_path)

                # ## Save modified tfrecord
                # output_tfrecord_path = f'hd_map_mod_{scene_index}_{frame_index}.tfrecord'
                # # save_tfrecord(frame_mod, map_features_list_mod, output_tfrecord_path)

                frame_index += 1
