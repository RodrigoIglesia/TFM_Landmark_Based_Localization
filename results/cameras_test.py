"""
Este módulo contiene las funciones necesarias para
 - paresear información del Waymo dataset
 - extraer las imágenes de las cámaras
"""

import os
import sys
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()


# Add project root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

from waymo_utils.WaymoParser import *



if __name__ == "__main__":
    scene = "individual_files_validation_segment-10289507859301986274_4200_000_4220_000_with_camera_labels"
    scene_path = os.path.join(src_dir, "dataset/final_tests_scene/" + scene + ".tfrecord")
    points = []

    frame_idx = 0
    image_idx = 0
    for frame in load_frame(scene_path):
        print(frame.timestamp_micros)

        # Read the 5 cammera images of the frame
        cameras_images = {}
        cameras_bboxes = {}
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)
            cameras_images[image.name] = decoded_image[...,::-1]
            cameras_bboxes[image.name] = get_image_bboxes(frame.camera_labels, image.name)
        ordered_images = order_camera(cameras_images, [4, 2, 1, 3, 5])
        ordered_bboxes = order_camera(cameras_bboxes, [4, 2, 1, 3, 5])

        canvas = generate_canvas(ordered_images)

        # Show cameras images
        cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Canvas", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
