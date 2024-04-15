import os
import sys
import pathlib
import datetime
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

tf.compat.v1.enable_eager_execution()

current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)


class WaymoYOLOConverter():
    def __init__(self,
                 image_dir,
                 label_dir,
                 image_prefix=None,
                 write_image=True,
                 add_waymo_info=False,
                 add_yolo_info=True,
                 frame_index_ones_place=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_prefix = image_prefix
        self.write_image = write_image
        self.add_waymo_info = add_waymo_info
        self.add_yolo_info = add_yolo_info
        if frame_index_ones_place is not None:
            self.frame_index_ones_place = int(frame_index_ones_place)
            assert 0 <= self.frame_index_ones_place < 10
        else:
            self.frame_index_ones_place = None

        self.init_waymo_dataset_proto_info()
        self.img_index = 0

    def init_waymo_dataset_proto_info(self):
        self.waymo_class_mapping = [
            'TYPE_UNKNOWN', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN',
            'TYPE_CYCLIST'
        ]

    def process_sequences(self, tfrecord_paths):
        if not isinstance(tfrecord_paths, (list, tuple)):
            tfrecord_paths = [tfrecord_paths]

        for tfrecord_index, tfrecord_path in enumerate(sorted(tfrecord_paths)):
            sequence_data = tf.data.TFRecordDataset(str(tfrecord_path),
                                                    compression_type='')

            for frame_index, frame_data in enumerate(sequence_data):
                if self.frame_index_ones_place is not None:
                    if frame_index % 10 != self.frame_index_ones_place:
                        continue
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(frame_data.numpy()))
                self.process_frame(frame, frame_index, tfrecord_index)

    def process_frame(self, frame, frame_index, tfrecord_index):
        print(f'frame: {frame_index}, {len(frame.images)} cameras')

        for camera_image in frame.images:
            self.process_img(camera_image, frame, frame_index, tfrecord_index)
            for camera_label in frame.camera_labels:
                if camera_label.name != camera_image.name:
                    continue
                self.add_yolo_annotation_dict(camera_image, camera_label, frame, tfrecord_index, frame_index)
            self.img_index += 1

    def process_img(self, camera_image, frame, frame_index, tfrecord_index):
        img_filename = f'{tfrecord_index:05d}_{frame_index:05d}_camera{camera_image.name}.jpg'  # noqa
        # img_filename = f'{frame.context.name}_{camera_image.name}_{frame.timestamp_micros}.jpg'  # noqa
        if self.image_prefix is not None:
            img_filename = self.image_prefix + '_' + img_filename
        img_path = os.path.join(self.image_dir, img_filename)

        img = tf.image.decode_jpeg(camera_image.image).numpy()[:, :, ::-1]
        img_height = img.shape[0]
        img_width = img.shape[1]
        if self.write_image:
            with open(img_path, 'wb') as f:
                f.write(bytearray(camera_image.image))

    def add_yolo_annotation_dict(self, camera_image, camera_label, frame, tfrecord_index, frame_index):
        for box_label in camera_label.labels:
            category_id = box_label.type

            width = box_label.box.length
            height = box_label.box.width
            x_center = box_label.box.center_x
            y_center = box_label.box.center_y

            # Normalize coordinates and dimensions
            x_center /= 1920
            y_center /= 1280
            width /= 1920
            height /= 1280

            annotation_dict = {
                "category_id": category_id,
                "bbox": [x_center, y_center, width, height],
                "iscrowd": 0,
            }

            if self.add_waymo_info:
                annotation_dict["track_id"] = box_label.id
                annotation_dict["det_difficult"] = \
                    box_label.detection_difficulty_level
                annotation_dict["track_difficult"] = \
                    box_label.tracking_difficulty_level

            label_filename = f"{tfrecord_index:05d}_{frame_index:05d}_camera{camera_image.name}.txt"
            if self.image_prefix is not None:
                label_filename = self.image_prefix + '_' + label_filename
            label_path = os.path.join(self.label_dir, label_filename)

            with open(label_path, mode='w') as f:
                line = (
                    f"{annotation_dict['category_id']} "
                    f"{annotation_dict['bbox'][0]} "
                    f"{annotation_dict['bbox'][1]} "
                    f"{annotation_dict['bbox'][2]} "
                    f"{annotation_dict['bbox'][3]}\n"
                )
                f.write(line)

def main():
    tfrecord_dir = src_dir + '/dataset/waymo_map_scene_mod'
    image_dir = src_dir + '/dataset/waymo_yolo_format/images/train'
    label_dir = src_dir + '/dataset/waymo_yolo_format/labels/train'

    tfrecord_list = list(
        sorted(pathlib.Path(tfrecord_dir).glob('*.tfrecord')))

    waymo_converter = WaymoYOLOConverter(image_dir, label_dir)
    waymo_converter.process_sequences(tfrecord_list)

if __name__ == "__main__":
    main()
