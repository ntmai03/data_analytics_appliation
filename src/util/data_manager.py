import sys
import os
from io import StringIO
import json
import csv
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cf


def load_csv_data(path):
    file_path = os.path.join(path)

    return pd.read_csv(file_path)


def s3_list_bucket():
    # Fetch the list of existing buckets
    clientResponse = cf.S3_CLIENT.list_buckets()

    print('Printing bucket names...')
    for bucket in clientResponse['Buckets']:
        print(f'Bucket Name: {bucket["Name"]}')


def s3_load_csv(bucket, filename):
    # Create the S3 object
    obj = cf.S3_CLIENT.get_object(
        Bucket = bucket,
        Key = filename
    )   
    # Read data from the S3 object
    return pd.read_csv(obj['Body'])



# To check whether root bucket exists or not
def bucket_exists(bucket_name, s3_resource):
    try:
        s3_resource.meta.client.head_bucket(Bucket=bucket_name)
        # print("Bucket exists.", bucket_name)
        exists = True
    except ClientError as error:
        error_code = int(error.response['Error']['Code'])
        if error_code == 403:
            print("Private Bucket. Forbidden Access! ", bucket_name)
        elif error_code == 404:
            #print("Bucket Does Not Exist!", bucket_name)
            exists = False
    
    return exists


# create a new bucket on S3
def create_s3_bucket(bucket_name, s3_resource):
    results = s3_resource.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': cf.S3_LOCATION_CONSTRAINT}
    )


# store a json file
def write_json_file(bucket_name, file_name, data, type='s3'):
    if(type == 's3'):
        cf.S3_CLIENT.put_object(Bucket=bucket_name, Key=file_name, Body = json.dumps(data).encode('UTF-8'))        
    elif(type == 'local'):
        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)


# write a csv file
def write_csv_file(bucket_name, file_name, data, type='s3'):
    if(type == 's3'):
        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False)
        cf.S3_CLIENT.put_object(Bucket=bucket_name, Key=file_name, Body = csv_buffer.getvalue())
    elif(type == 'local'):
        data.to_csv(file_name, index=False)


# read a csv file
def read_csv_file(bucket_name, file_name, type='s3'):
    if(type == 's3'):
        data = cf.S3_CLIENT.get_object(Bucket=cf.S3_DATA_PATH, Key=file_name)
        data = pd.read_csv(data['Body'])
    elif(type == 'local'):
        data = pd.read_csv(file_name)
    return data


def split_data(X, y, test_size=0.2, random_state=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
    