import os
import sys
import pathlib
import json

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

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
        figure = plot_maps.plot_map_features(map_features)
        figure.show()