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
import xml.etree.ElementTree as ET

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


def build_s3_hierarchy(objects):
    hierarchy = ET.Element("S3Hierarchy")

    for obj in objects:
        key_parts = obj['Key'].split('/')
        current_node = hierarchy

        for part in key_parts:
            existing_node = next((child for child in current_node.findall(part) if child.tag == part), None)

            if existing_node is not None:
                current_node = existing_node
            else:
                new_node = ET.SubElement(current_node, part)
                current_node = new_node

    return hierarchy

def write_hierarchy_to_xml(node, file_path):
    tree = ET.ElementTree(node)
    tree.write(file_path)



if __name__ == "__main__":
    # Create an S3 client with anonymous credentials
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Specify the S3 bucket name
    bucket_name = 'avg-kitti'

    # List objects in the specified S3 bucket
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    objects = list_objects(s3, bucket_name, 'raw_data')
    hierarchy = build_s3_hierarchy(objects)


    output_file_path = "S3_hierarchy.xml"
    # Clear the existing content of the file before writing the new hierarchy

    with open(output_file_path, 'w') as file:
        file.write('')

    write_hierarchy_to_xml(hierarchy, output_file_path)
    print(f'S3 hierarchy written to {output_file_path}')
        