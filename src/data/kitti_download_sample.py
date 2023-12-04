"""
SRC code for using KITTI dataset stored on AWS S3 public bucket
This script uses boto3 library to make requests to the S3
KITTI S3 using CLI: aws s3 ls --no-sign-request s3://avg-kitti/
"""

import cv2
import boto3
from botocore import UNSIGNED
from botocore.client import Config

import zipfile

import sys
import os

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_script_directory)


if __name__ == "__main__":
    # Create an S3 client with anonymous credentials
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Specify the S3 bucket name
    bucket_name = 'avg-kitti'
    prefix = 'raw_data'

    sample_data = "raw_data/2011_09_28_drive_0080/2011_09_28_drive_0080_sync.zip"
    local_file = "2011_09_28_drive_0080_sync.zip"

    # Downlad S3 sample file
    zip_file_path = os.path.join(current_script_directory, local_file)
    print(zip_file_path)
    # Check if file already exists
    if os.path.exists(zip_file_path):
        print(f'The file {zip_file_path} already exists. Skipping download.')
    else:
        s3.download_file(bucket_name, sample_data, zip_file_path)

        # Extract downloaded zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(current_script_directory + "/samples/")

        # Delete zip file
        os.remove(zip_file_path)

    # Navigate the unziped folder
    