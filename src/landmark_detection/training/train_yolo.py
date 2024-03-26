import os
import sys

from ultralytics import YOLO

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)

# Load a model
model = YOLO('models/yolov8s.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='data.yaml', epochs=100, imgsz=640)