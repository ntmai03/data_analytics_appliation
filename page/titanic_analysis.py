import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os
from io import BytesIO

sys.path.append('src')
from src import config as cf
from src.util import data_manager as dm
from analysis.titanic_ds import Titanic

def app():
	#st.sidebar.subheader('Select function')
	st.sidebar.subheader('')
	st.sidebar.subheader('')
	task_type = ['Select Function',
				 'Data Snapshot',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Predictive Model',
				 'Prediction']
	task_option = st.sidebar.selectbox('', task_type)


	if task_option == 'Select Function':
		st.write('Introduction')


		cat = ["bored", "happy", "bored", "bored", "happy", "bored"]
		dog = ["happy", "happy", "happy", "happy", "bored", "bored"]
		activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

		width = st.sidebar.slider("plot width", 0.1, 25., 3.)
		height = st.sidebar.slider("plot height", 0.1, 25., 1.)

		fig, ax = plt.subplots(figsize=(width, height))
		ax.plot(activity, dog, label="dog")
		ax.plot(activity, cat, label="cat")
		ax.legend()

		buf = BytesIO()
		fig.savefig(buf, format="png")
		st.image(buf)