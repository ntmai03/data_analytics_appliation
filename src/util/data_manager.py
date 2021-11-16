import sys
import os
from pathlib import Path
import boto3

import pandas as pd
# split data
from sklearn.model_selection import train_test_split

#sys.path.append('src')

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


def split_data(X, y, test_size=0.2, random_state=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
    