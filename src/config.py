import streamlit as st
import sys
import os
from pathlib import Path
# import boto3 - AWS SDK for python provided by Amazon
import boto3

sys.path.append('src')

# Define path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config.yml')
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
TRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')
PIPELINE_PATH = os.path.join(SRC_PATH, 'pipeline')

S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id = st.secrets["aws_access_key_id"],
    aws_secret_access_key = st.secrets["aws_secret_access_key"],
    region_name = 'us-east-2'
)
S3_DATA_PATH = 'datool-data'
S3_DATA_RAW_PATH = 'raw/'



def test_funct():
	st.write('test')

"""
S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id = st.secrets["aws_access_key_id"],
    aws_secret_access_key = st.secrets["aws_secret_access_key"],
    region_name = st.secrets["region_name"]
)

aws_access_key_id = "AKIAZFGIKEZ3JECBN5MV"
aws_secret_access_key = "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"
region_name = "us-east-2"
"""
