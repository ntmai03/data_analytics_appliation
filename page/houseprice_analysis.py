import streamlit as st
import numpy as np
import pandas as pd
import joblib

import sys
assert sys.version_info >= (3, 5)
from pathlib import Path
import os


PROJECT_ROOT_DIR =  Path(__file__).resolve().parents[1]
# source code
FUNCTION_PATH = os.path.join(PROJECT_ROOT_DIR, "src")
sys.path.append(FUNCTION_PATH)

from src import config as cf
from src.util import data_manager as dm

def app():
	st.sidebar.subheader('Select function')
	task_type = ['Introduction',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Predictive Model',
				 'Prediction']
	task_option = st.sidebar.selectbox('', task_type)
	st.sidebar.header('')

	if task_option == 'Introduction':
		st.write(cf.DATA_RAW_PATH)
		data_file = os.path.join(cf.DATA_RAW_PATH, "kc_house_data.csv")
		df = dm.load_csv_data(data_file)
		st.write("#### First 100 rows")
		st.write(df.head(100))
