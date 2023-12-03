from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from google.cloud import storage

import numpy as np
import pandas as pd

storage_client = storage.Client()
bucket_name = "waymo_open_dataset_v_2_0_0"

bucket = storage_client.bucket(bucket_name)

blob = bucket.get_blob('validation/camera_image/10203656353524179475_7625_000_7645_000.parquet')

# Imprime el nombre del archivo
print(blob.name)

# Imprime el tamaño del archivo
print(blob.size)

# Descarga el archivo
blob.download_to_filename('validation/camera_image/10203656353524179475_7625_000_7645_000.parquet')


