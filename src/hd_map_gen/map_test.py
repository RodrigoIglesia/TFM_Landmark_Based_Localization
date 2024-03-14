import os
import sys
import pathlib
import json

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()
from google.protobuf.json_format import MessageToJson

# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(src_dir)

from src.waymo_utils.WaymoParser import *
from src.waymo_utils.waymo_3d_parser import *

if __name__ == "__main__":
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")

    tfrecord_list = list(
        sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))
    print('-------------------------------------\n')
    print('-------------------------------------\n')

    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        for frame in load_frame(scene_path):
            # Copy the frame object to modify it
            frame_mod = frame
            if hasattr(frame, 'map_features') and frame.map_features:
                # Retrieve map_feature in the firts frame
                map_features = frame.map_features
                # print(map_features)

                # Each map feature is a json, decode jsons sepparately and append them to a list, for each scene map
                map_features_list = []
                for map_item in map_features:
                    map_features_list.append(json.loads(MessageToJson(map_item)))


                ## Save json file of original map
                # Load list of jsons in a json
                file_path = 'hd_map' + str(scene_index) + '.json'

                with open(file_path, 'w') as json_file:
                    print('Saving: ', file_path)
                    json.dump(map_features_list, json_file)
                    json_file.write(str(map_features_list))

                # Modify the map adding new features (sign)
                sign_item_id = str(int(map_features_list[-1]["id"]) + 1)
                sign_json = {
                    "id":sign_item_id,
                             "sign": {
                                 "position": {
                                    "x": -1310.4264358044165,
                                    "y": 10557.175145829448,
                                    "z": 35.369245673324315
                                    }
                             }}

                # Save modified json
                map_features_list_mod = map_features_list
                map_features_list_mod.append(sign_json)
                file_path_mod = 'hd_map_mod' + str(scene_index) + '.json'

                # Use json.dump to save the dictionary to a JSON file
                with open(file_path_mod, 'w') as json_file:
                    print('Saving: ', file_path_mod)
                    json.dump(map_features_list_mod, json_file)