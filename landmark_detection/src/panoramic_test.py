import os
import sys
from waymo_open_dataset import dataset_pb2 as open_dataset

# Add project src root to python path
# Add project root root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.insert(0, src_dir)

from waymo_utils.WaymoParser import *
from waymo_utils.waymo_2d_parser import *



if __name__ == "__main__":
    # TODO: ruta al dataset tiene que ser configurable y única para todos los scripts
    dataset_path = os.path.join(src_dir, "dataset/waymo_map_scene")
    tfrecord_list = list(sorted(pathlib.Path(dataset_path).glob('*.tfrecord')))

    ##############################################################
    ## Iterate through Dataset Scenes
    ##############################################################
    for scene_index, scene_path in enumerate(sorted(tfrecord_list)):
        frame = next(load_frame(scene_path))
        print(frame.timestamp_micros)

        # Get the cameras calibration parameters for the frame
        camera_calibrations = frame.context.camera_calibrations

        # Read the 5 cammera images of the frame
        cameras_images = {}
        for i, image in enumerate(frame.images):
            decoded_image = get_frame_image(image)
            cameras_images[image.name] = decoded_image[...,::-1]

        # Custom order of keys
        custom_order = [4, 2, 1, 3, 5]

        # Create a list of tuples (key, image_data)
        key_image_tuples = [(key, cameras_images[key]) for key in cameras_images]
        # Sort the list of tuples based on the custom order
        sorted_tuples = sorted(key_image_tuples, key=lambda x: custom_order.index(x[0]))

        # Extract image data from sorted tuples
        ordered_images = [item[1] for item in sorted_tuples[1:4]]


        ############################################################################
        ## HOMOGRAPY
        ############################################################################
        front_intrinsic = np.array(camera_calibrations[0].intrinsic).reshape(3,3)
        front_extrinsic = np.array(camera_calibrations[0].extrinsic.transform).reshape(4,4)
        print(front_intrinsic)
        print(front_extrinsic)
        print("\n")
        front_corners = np.array([[0,0], [camera_calibrations[0].width, 0], [camera_calibrations[0].width, camera_calibrations[0].height], [0, camera_calibrations[0].height]])


        # Resize images
        # resized_images = reshape_images(ordered_images)

        # canvas = generate_canvas(ordered_images)


        # # Show cameras images
        # cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("Canvas", canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()