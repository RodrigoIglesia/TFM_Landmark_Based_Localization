"""
SRC code for using KITTI dataset stored on AWS S3 public bucket
This script uses boto3 library to make requests to the S3
KITTI S3 using CLI: aws s3 ls --no-sign-request s3://avg-kitti/
"""
import sys
import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(src_dir)

from utils import s3_utils


if __name__ == "__main__":
    # Scene to download
    scene_date  = "2011_09_26"
    scene_drive = "0001"
    scene_name  = scene_date + "_drive_" + scene_drive

    # Create directory for new scene
    dest_folder = current_script_directory + "/samples/"

    # Create an S3 client with anonymous credentials
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # S3 bucket name and folder to download
    bucket_name   = 'avg-kitti'
    folder_prefix = 'raw_data'
    scene_s3_path = folder_prefix + "/" + scene_name

    # Download calibration data
    dest_calib_zip = dest_folder + '/' + scene_date + "_calib.zip"
    ori_calib_zip  = folder_prefix + '/' + scene_date + "_calib.zip"
    s3_utils.download_s3_object(s3, bucket_name, dest_calib_zip, ori_calib_zip)
    s3_utils.extract_zip_file(dest_calib_zip, dest_folder)

    objects_list = s3_utils.list_objects(s3, bucket_name, scene_s3_path)
    for obj in objects_list:
        object_name = obj['Key']
        file_name = object_name.split('/')[-1]
        dest_file = file_name

        dest_file_path = os.path.join(dest_folder, dest_file)
        s3_utils.download_s3_object(s3, bucket_name, dest_file_path, object_name)

        if object_name[-3:] == 'zip':
            s3_utils.extract_zip_file(dest_file_path, dest_folder)