import os
import sys

from ultralytics.data.converter import convert_coco

# Add project src root to python path
# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)

coco_annot_path = src_dir + '/dataset/waymo_coco_format/test/coco_annotations'
yolo_annot_path = src_dir + '/dataset/waymo_coco_format/test/yolo_annotations'
convert_coco(labels_dir=coco_annot_path, save_dir=yolo_annot_path, use_segments=False, use_keypoints=False, cls91to80=True)