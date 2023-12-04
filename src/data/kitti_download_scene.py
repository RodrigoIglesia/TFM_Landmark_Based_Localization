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

def list_objects(client, bucket_name, prefix=''):
    objects = []

    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in result:
            objects.extend(result['Contents'])

    return objects



if __name__ == "__main__":
    # Scene to download
    scene_date  = "2011_09_26"
    scene_id    = "0001"
    scene_name = scene_date + "_drive_" + scene_id

    dest_folder = current_script_directory + "/samples/" + scene_name
    os.mkdir(dest_folder)

    # Create an S3 client with anonymous credentials
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'avg-kitti'
    prefix      = 'raw_data'

    scene_s3_path = prefix + "/" + scene_name

    objects = list_objects(s3, bucket_name, scene_s3_path)
    for obj in objects:
        object_name = obj['Key']
        file_name = object_name.split('/')[-1]
        dest_file = file_name

        # Downlad S3 file
        dest_file_path = os.path.join(dest_folder, dest_file)
        if os.path.exists(dest_file_path):
            print(f'The file {dest_file_path} already exists. Skipping download.')
        else:
            s3.download_file(bucket_name, object_name, dest_file_path)
            print(f"S3 object downloaded: {dest_file_path}")

        if object_name[-3:] == 'zip':
            # Extract downloaded zip file
            with zipfile.ZipFile(dest_file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                zip_ref.extractall(dest_folder)

            # Delete zip file
            os.remove(dest_file_path)

