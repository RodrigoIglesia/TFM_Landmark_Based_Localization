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


def print_s3_hierarchy(objects, indent=2):
    hierarchy = {}

    for obj in objects:
        key_parts = obj['Key'].split('/')
        node = hierarchy

        for part in key_parts:
            node = node.setdefault(part, {})

    def print_node(node, level=0):
        for key, value in node.items():
            print('    ' * (level * indent) + f'{key}/')
            print_node(value, level + 1)

    print_node(hierarchy)


if __name__ == "__main__":
    # Create an S3 client with anonymous credentials
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Specify the S3 bucket name
    bucket_name = 'avg-kitti'

    # List objects in the specified S3 bucket
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    objects = []

    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in result:
            objects.extend(result['Contents'])
    print_s3_hierarchy(objects)
        