import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pykitti

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

dataset_path = src_dir + "/data/samples"
scene_date   = "2011_09_26"
scene_drive  = "0001"

dataset = pykitti.raw(dataset_path, scene_date, scene_drive)

### Read Velodyne scans ###
for velo_scan in dataset.velo:
    print(velo_scan)