class WaymoParser:
    def __init__(self):
        pass

    def load_frame(scene):
        """
        Load and yields frame object of a determined scene
        A frame is composed of imagrs from the 5 cameras
        A frame also has information of the bounding boxes and labels, related to each image
        Args: scene (str) - path to the scene which contains the frames
        Yield: frame_object (dict) - frame object from waymo dataset containing cameras and laser info
        """
        dataset = tf.data.TFRecordDataset(scene, compression_type='')
        for data in dataset:
            frame_object = open_dataset.Frame()
            frame_object.ParseFromString(bytearray(data.numpy()))

            yield frame_object

    def get_frame_image(frame_image):
        """
        Decodes a single image contained in the frame
        Args: frame_image - image in waymo format
        Return: decoded_image - Numpy decoded image
        """
        decoded_image = tf.image.decode_jpeg(frame_image.image)
        decoded_image = decoded_image.numpy()

        return decoded_image


class Waymo2DParser(WaymoParser):
    def __init__(self):
        pass

class WaymoSemanticParser(WaymoParser):
    def __init__(self):
        pass

class Waymo3DParser(WaymoParser):
    def __init__(self):
        pass