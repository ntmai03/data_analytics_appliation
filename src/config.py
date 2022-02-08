import streamlit as st
import sys
import os
from pathlib import Path
# import boto3 - AWS SDK for python provided by Amazon
import boto3
import yaml

sys.path.append('src')

# Define path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config.yml')
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
TRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')
DATA_PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
PIPELINE_PATH = os.path.join(SRC_PATH, 'pipeline')
ANALYSIS_PATH = os.path.join(SRC_PATH, 'analysis')
# image path
IMAGE_PATH = os.path.join(PROJECT_ROOT, 'image')

S3_LOCATION_CONSTRAINT = 'us-east-2'
S3_DATA_PATH = 'datool-data'
S3_DATA_RAW_PATH = 'raw/'
S3_DATA_PROCESSED_PATH = 'processed/'

# Creating the low level functional client
S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id = st.secrets["aws_access_key_id"],
    aws_secret_access_key = st.secrets["aws_secret_access_key"],
    region_name = st.secrets["region_name"]
)

S3_RESOURCE = boto3.resource(
    's3',
    aws_access_key_id = st.secrets["aws_access_key_id"],
    aws_secret_access_key = st.secrets["aws_secret_access_key"],
    region_name = st.secrets["region_name"] 
)

"""
CONFIF FOR HOTEL RECOMMENDATION APP
"""
# AWS S3
S3_DATA_BOOKING = 'booking'
S3_BOOKING_REVIEW = 'review'
S3_BOOKING_HOTEL = 'hotel'
HOTEL_LIST_FILE = 'booking_list.csv'

# BOOKING_SEARCH_URL
BOOKING_SEARCH_HOTEL = 'https://booking-com.p.rapidapi.com/v1/hotels/search'
BOOKING_SEARCH_LOCATION = 'https://booking-com.p.rapidapi.com/v1/hotels/locations'
BOOKING_RAPIDAPID_QUERYSTRING = {
            'x-rapidapi-key': st.secrets["rapidapid_booking_key"],
            'x-rapidapi-host': "booking-com.p.rapidapi.com"
        }



def fetch_config_from_yaml(cfg_path: Path = CONFIG_FILE_PATH) -> yaml:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = yaml.load(conf_file, Loader=yaml.FullLoader)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def update_yaml_config_file(config, cfg_path: Path = CONFIG_FILE_PATH):
    with open(cfg_path, 'w') as yamlfile:
        yaml.dump(config, yamlfile)
        print("Write successful")    


data = fetch_config_from_yaml()
