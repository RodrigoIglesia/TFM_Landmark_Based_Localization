from google.protobuf.json_format import Parse
from your_protobuf_module import YourProtobufMessage
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
    # Specify the file path of the JSON file
    file_path = 'hd_map.json'

    # Open the JSON file and load the data
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Now 'data' contains the contents of the JSON file
    print("Data loaded from JSON file:")
    print(data)

    # Parse the JSON string into a Protobuf message
    protobuf_message = YourProtobufMessage()
    Parse(data, protobuf_message)

    # Access the repeated field (assuming it's a RepeatedCompositeCo)
    repeated_field = protobuf_message.your_protobuf_field

    # Now, repeated_field is an instance of RepeatedCompositeCo containing your data
    print(repeated_field)
