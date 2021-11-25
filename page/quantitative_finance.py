'''
Configuration
'''
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
from pathlib import Path
import os

'''
Library
'''
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# utilities
from pages import utils

# project root
PROJECT_ROOT_DIR=  Path(__file__).resolve().parents[1]
# source code
FUNCTION_PATH = os.path.join(PROJECT_ROOT_DIR, "src")
# image path
IMAGE_PATH = os.path.join(PROJECT_ROOT_DIR, "image")
# data path
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data")
DATA_RAW = os.path.join(DATA_PATH, "raw")
DATA_PROCESSED = os.path.join(DATA_PATH, "processed")
# model path
MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, "model")

sys.path.append(FUNCTION_PATH)

# Visualization
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


from util import (
    data_manager as dm,
    data_exploration as de)
# from pipeline.shopping_recommendation_pipeline import *

# Supress warnings
import warnings
warnings.filterwarnings("ignore")


def app():

    st.sidebar.subheader('Select function')
    task_type = ["Introduction", 
                 "Exploratory Data Analysis",
                 "Data Processing",
                 "Predictive Model",
                 "Prediction"]
    task_option = st.sidebar.selectbox('',task_type)
    st.sidebar.header('')

    if task_option == 'Introduction':
        st.wtrite('https://www.udemy.com/course/credit-risk-modeling-in-python/learn/lecture/15585744#overview
        st.write('https://www.udemy.com/course/quantitative-finance-algorithmic-trading-in-python/learn/lecture/14002560#overview')
        st.write('https://www.udemy.com/course/python-for-finance-and-trading-algorithms/learn/lecture/7054788#overview')
        st.write('https://www.udemy.com/course/the-complete-financial-analyst-course/learn/lecture/4281754#overview')

    if task_option == 'Exploratory Data Analysis':
        st.write("#### Comming soon...")

    if task_option == 'Data Processing':
        st.write("#### Comming soon...")

    if task_option == 'Predictive Model':
        st.write("#### Comming soon...")

    if task_option == 'Prediction':
        st.write("#### Input your data for prediction")
