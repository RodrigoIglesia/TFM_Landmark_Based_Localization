import sys
import os
import zipfile

import boto3
from botocore import UNSIGNED
from botocore.client import Config

def list_objects(client, bucket, prefix=''):
    objects = []

    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in result:
            objects.extend(result['Contents'])

    return objects


def download_s3_object(client, bucket, destination, obj_name):
    if os.path.exists(destination):
        print(f'The file {destination} already exists. Skipping download.')
    else:
        client.download_file(bucket, obj_name, destination)
        print(f"S3 object downloaded: {destination}")


def extract_zip_file(zip_file_path, destination):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    # Delete zip file
    os.remove(zip_file_path)